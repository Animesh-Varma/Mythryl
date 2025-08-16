import os
import sys
import re
import csv
import glob
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_CHATS_DIR = os.path.join(PROJECT_ROOT, 'extracted_chats')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
OUTPUT_CSV_PATH = os.path.join(TEMP_DIR, 'persona_style_v2.csv')
DB_PATH = os.path.join(TEMP_DIR, 'my_style_v2.index')
TEXT_COLUMN = 'prompt'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Helper Functions ---

def get_persona_from_path(file_path):
    """Extracts a persona from the chat file path."""
    try:
        parent_dir = os.path.dirname(file_path)
        dir_name = os.path.basename(parent_dir)

        if dir_name in ['Personal', 'Group']:
            return os.path.splitext(os.path.basename(file_path))[0]
        else:
            return dir_name
    except Exception:
        return None


def create_context_dataset(chat_files, output_csv_file, user_sender_name):
    """Creates a comprehensive dataset handling conversation nuances."""
    from datetime import datetime, timedelta

    line_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s[AP]M) - ([^:]+): (.*)")
    all_training_examples = []

    print(f"Processing {len(chat_files)} chat files with improved context handling...")

    for chat_file in chat_files:
        persona = get_persona_from_path(chat_file)
        if not persona: continue

        with open(chat_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse all messages with timestamps
        raw_messages = []
        for line in lines:
            match = line_pattern.match(line)
            if match:
                timestamp_str = match.group(1)
                sender = match.group(2).strip()
                content = match.group(3).strip()

                if content in ["<Media omitted>", "You deleted this message"] or not content:
                    continue

                # Parse timestamp for conversation breaks
                timestamp = None
                for fmt in ["%m/%d/%y, %I:%M %p", "%m/%d/%Y, %I:%M %p", "%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p"]:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except:
                        continue

                raw_messages.append({
                    "timestamp": timestamp,
                    "sender": sender,
                    "content": content,
                    "persona": persona
                })

        if not raw_messages:
            continue

        # Group consecutive messages by same sender
        consolidated_messages = []
        i = 0
        while i < len(raw_messages):
            current_sender = raw_messages[i]["sender"]
            messages_group = [raw_messages[i]["content"]]
            start_time = raw_messages[i]["timestamp"]

            # Collect consecutive messages from same sender
            j = i + 1
            while j < len(raw_messages) and raw_messages[j]["sender"] == current_sender:
                # Check for conversation breaks (more than 2 hours gap)
                if (start_time and raw_messages[j]["timestamp"] and
                        raw_messages[j]["timestamp"] - start_time > timedelta(hours=2)):
                    break
                messages_group.append(raw_messages[j]["content"])
                j += 1

            consolidated_messages.append({
                "sender": current_sender,
                "content": " [MSG_BREAK] ".join(messages_group),
                "timestamp": start_time,
                "persona": persona,
                "message_count": len(messages_group)
            })
            i = j

        # Create diverse training examples
        for i, msg in enumerate(consolidated_messages):
            if msg["sender"] == user_sender_name:

                # 1. CONVERSATION INITIATION: User starts new conversation
                if i == 0 or (msg["timestamp"] and consolidated_messages[i - 1]["timestamp"] and
                              msg["timestamp"] - consolidated_messages[i - 1]["timestamp"] > timedelta(hours=4)):
                    all_training_examples.append({
                        "prompt": "[CONVERSATION_START]",
                        "response": msg["content"],
                        "persona": msg["persona"],
                        "type": "initiation"
                    })

                # 2. CONTEXT-AWARE RESPONSES: User responds with context
                else:
                    context_messages = []
                    context_limit = 4  # Number of previous messages

                    # Collect relevant context
                    start_idx = max(0, i - context_limit)
                    for j in range(start_idx, i):
                        prev_msg = consolidated_messages[j]
                        # Skip very old messages (more than 6 hours)
                        if (msg["timestamp"] and prev_msg["timestamp"] and
                                msg["timestamp"] - prev_msg["timestamp"] > timedelta(hours=6)):
                            continue
                        context_messages.append(f"{prev_msg['sender']}: {prev_msg['content']}")

                    if context_messages:
                        prompt = " [TURN_BREAK] ".join(context_messages)
                        all_training_examples.append({
                            "prompt": prompt,
                            "response": msg["content"],
                            "persona": msg["persona"],
                            "type": "contextual_response"
                        })

                # 3. DIRECT Q&A PAIRS: Immediate response to questions
                if i > 0:
                    prev_msg = consolidated_messages[i - 1]
                    if prev_msg["sender"] != user_sender_name:
                        # Check if previous message seems like a question or prompt
                        prev_content = prev_msg["content"].lower()
                        question_indicators = [
                            "?", "what", "how", "when", "where", "why", "who",
                            "can you", "could you", "would you", "will you",
                            "do you", "did you", "have you", "are you",
                            "tell me", "explain", "describe", "think about"
                        ]

                        if any(indicator in prev_content for indicator in question_indicators):
                            all_training_examples.append({
                                "prompt": prev_msg["content"],
                                "response": msg["content"],
                                "persona": msg["persona"],
                                "type": "direct_qa"
                            })

            # 4. TOPIC TRANSITIONS: When user changes subjects
            if (msg["sender"] == user_sender_name and i > 1 and
                    consolidated_messages[i - 1]["sender"] != user_sender_name):

                # Look for topic shifts (user introducing new topics after others)
                topic_starters = [
                    "by the way", "btw", "also", "oh", "speaking of", "that reminds me",
                    "guess what", "did you hear", "i just", "just saw", "just heard"
                ]

                content_lower = msg["content"].lower()
                if any(starter in content_lower for starter in topic_starters):
                    prev_context = consolidated_messages[i - 1]["content"]
                    all_training_examples.append({
                        "prompt": f"[TOPIC_SHIFT] Previous: {prev_context}",
                        "response": msg["content"],
                        "persona": msg["persona"],
                        "type": "topic_transition"
                    })

    # Write enhanced CSV with metadata
    with open(output_csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response", "persona", "type"])

        for example in all_training_examples:
            writer.writerow([
                example["prompt"],
                example["response"],
                example["persona"],
                example["type"]
            ])

    # Print comprehensive statistics
    total_examples = len(all_training_examples)
    stats = {}
    for example in all_training_examples:
        stats[example["type"]] = stats.get(example["type"], 0) + 1

    print(f"Successfully created enhanced dataset: {output_csv_file}")
    print(f"Total training examples: {total_examples}")
    for example_type, count in stats.items():
        percentage = (count / total_examples) * 100
        print(f"  - {example_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

def create_vector_database(csv_path, text_column, db_path, model_name):
    """Creates a vector database from a CSV file."""
    try:
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    df.dropna(subset=[text_column], inplace=True)
    df = df[df[text_column].str.strip() != '']

    if df.empty:
        print("Error: No text data found in the specified column after cleaning.")
        return

    texts = df[text_column].tolist()
    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Encoding text into vectors... (This may take a while)")
    embeddings = model.encode(texts, show_progress_bar=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    print("Adding vectors to the FAISS index...")
    index.add(embeddings)
    print(f"Saving FAISS index to {db_path}...")
    faiss.write_index(index, db_path)
    print(f"Successfully created vector database with {index.ntotal} vectors.")


def validate_sender_name(chat_files, sender_name):
    """Validates that the sender name exists in the chat files."""
    line_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s[AP]M) - ([^:]+): (.*)")
    found_senders = set()

    print("Validating sender name...")
    for chat_file in chat_files[:3]:  # Check first few files for efficiency
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                for line in f.readlines()[:50]:  # Check first 50 lines
                    match = line_pattern.match(line)
                    if match:
                        found_senders.add(match.group(2).strip())
        except Exception:
            continue

    if sender_name in found_senders:
        print(f"✓ Sender name '{sender_name}' found in chat files.")
        return True
    else:
        print(f"✗ Sender name '{sender_name}' not found in chat files.")
        print(f"Found these sender names: {', '.join(sorted(found_senders))}")
        return False

def save_sender_name(sender_name, temp_dir):
    """Saves the sender name to a text file in temp directory."""
    sender_file_path = os.path.join(temp_dir, 'sender_name.txt')
    with open(sender_file_path, 'w', encoding='utf-8') as f:
        f.write(sender_name)
    print(f"✓ Sender name saved to {sender_file_path}")

def display_persona_statistics(chat_files):
    """Displays statistics about personas found in chat files."""
    personas = set()
    personal_files = 0
    group_files = 0

    for file_path in chat_files:
        persona = get_persona_from_path(file_path)
        if persona:
            personas.add(persona)
            if 'Personal' in file_path:
                personal_files += 1
            else:
                group_files += 1

    print(f"\n--- Dataset Statistics ---")
    print(f"Total personas found: {len(personas)}")
    print(f"Personal chat files: {personal_files}")
    print(f"Group chat files: {group_files}")
    print(f"Personas: {', '.join(sorted(personas))}")
    print("-------------------------\n")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Chatbot Setup ---")

    # 1. Check and create directories
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    if not os.path.exists(EXTRACTED_CHATS_DIR):
        os.makedirs(os.path.join(EXTRACTED_CHATS_DIR, 'Personal'))
        os.makedirs(os.path.join(EXTRACTED_CHATS_DIR, 'Group'))
        print(f"IMPORTANT: The directory structure has been created for you in '{EXTRACTED_CHATS_DIR}'.")
        print("Please place your chat files in the appropriate folders:")
        print("  - Personal chats: Place individual .txt files or create subfolders in 'Personal/'")
        print("  - Group chats: Place individual .txt files or create subfolders in 'Group/'")
        print("  - Each subfolder will be treated as a separate persona")
        print("  - Direct .txt files in Personal/Group will also be treated as individual personas")

        input("\nPress Enter after you have placed your chat files in the folders...")

    # 2. Find chat files
    personal_chats = glob.glob(os.path.join(EXTRACTED_CHATS_DIR, 'Personal', '**', '*.txt'), recursive=True)
    group_chats = glob.glob(os.path.join(EXTRACTED_CHATS_DIR, 'Group', '**', '*.txt'), recursive=True)
    all_chat_files = personal_chats + group_chats

    if all_chat_files:
        display_persona_statistics(all_chat_files)

    if not all_chat_files:
        print("WARNING: No chat files found in the 'extracted_chats/Personal' or 'extracted_chats/Group' directories.")
        sys.exit(0)

    sender_name = input("Enter your sender name as it appears in the chat files: ").strip()
    if not sender_name:
        print("Error: Sender name cannot be empty.")
        sys.exit(1)

    if not validate_sender_name(all_chat_files, sender_name):
        retry = input("Would you like to try a different name? (y/n): ").lower()
        if retry == 'y':
            sender_name = input("Enter your sender name: ").strip()
            if not validate_sender_name(all_chat_files, sender_name):
                print("Sender name still not found. Exiting.")
                sys.exit(1)
        else:
            sys.exit(1)

    save_sender_name(sender_name, TEMP_DIR)

    # 3. Generate dataset and vector database
    create_context_dataset(all_chat_files, OUTPUT_CSV_PATH, sender_name)
    create_vector_database(OUTPUT_CSV_PATH, TEXT_COLUMN, DB_PATH, MODEL_NAME)

    print("--- Setup Complete ---")
    print("You can now run the chatbot using: python gemini_chat_final.py")