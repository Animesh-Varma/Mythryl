# setup.py

"""
This script prepares the data for the chatbot. It performs the following steps:
1.  Reads chat export files from the 'extracted_chats' directory.
2.  Processes the chat files to create a structured dataset of prompts and responses.
    It identifies different types of interactions like conversation starters, Q&A, etc.
3.  Saves the structured data into a CSV file ('persona_style_v2.csv').
4.  Uses a sentence transformer model to create vector embeddings for the prompts.
5.  Saves these embeddings into a FAISS index ('style_v2.index') for fast similarity search.
"""

import os
import sys
import re
import csv
import glob
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import zipfile

# --- Configuration ---
_PREFIX_RE = re.compile(r'^(?:whatsapp\s+chat\s+with\s+)', re.IGNORECASE)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_CHATS_DIR = os.path.join(PROJECT_ROOT, 'extracted_chats')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
OUTPUT_CSV_PATH = os.path.join(TEMP_DIR, 'persona_style_v2.csv')
DB_PATH = os.path.join(TEMP_DIR, 'style_v2.index')
TEXT_COLUMN = 'prompt'
MODEL_NAME = 'all-MiniLM-L6-v2'


def get_persona_from_path(file_path):
    """
    Extracts a persona name from the chat file path.
    - If the chat is in 'Personal' or 'Group', the persona is the filename.
    - Otherwise, the persona is the name of the parent directory.
    """
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
    """
    Processes raw chat logs into a structured CSV dataset of prompts and responses.

    This function reads through chat files, groups messages, and creates several
    types of training examples to capture different conversational styles:
    - Initiations: Starting a new conversation.
    - Contextual Responses: Replies within an ongoing conversation.
    - Direct Q&A: Answering a direct question.
    - Topic Transitions: Changing the subject.

    Special tokens are used to structure the data:
    - '[MSG_BREAK]': Joins multiple messages from the same sender.
    - '[TURN_BREAK]': Separates turns in a multi-turn context prompt.
    """
    line_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s[AP]M) - ([^:]+): (.*)")
    all_training_examples = []

    print(f"Processing {len(chat_files)} chat files with improved context handling...")

    for chat_file in chat_files:
        persona = get_persona_from_path(chat_file)
        if not persona:
            continue

        with open(chat_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 1. Parse all messages and their timestamps from the file.
        raw_messages = []
        for line in lines:
            match = line_pattern.match(line)
            if match:
                timestamp_str, sender, content = match.groups()
                sender = sender.strip()
                content = content.strip()

                if content in ["<Media omitted>", "You deleted this message"] or not content:
                    continue

                # Attempt to parse various common timestamp formats.
                timestamp = None
                for fmt in ["%m/%d/%y, %I:%M %p", "%m/%d/%Y, %I:%M %p", "%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p"]:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue

                raw_messages.append({
                    "timestamp": timestamp,
                    "sender": sender,
                    "content": content,
                    "persona": persona
                })

        if not raw_messages:
            continue

        # 2. Group consecutive messages from the same sender into single blocks.
        consolidated_messages = []
        i = 0
        while i < len(raw_messages):
            current_sender = raw_messages[i]["sender"]
            messages_group = [raw_messages[i]["content"]]
            start_time = raw_messages[i]["timestamp"]

            j = i + 1
            while j < len(raw_messages) and raw_messages[j]["sender"] == current_sender:
                # A gap of over 2 hours is considered a break in conversation.
                if (
                    start_time
                    and raw_messages[j]["timestamp"]
                    and raw_messages[j]["timestamp"] - start_time > timedelta(hours=2)
                ):
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

        # 3. Create diverse training examples from the consolidated messages.
        for i, msg in enumerate(consolidated_messages):
            if msg["sender"] == user_sender_name:

                # Example Type 1: CONVERSATION INITIATION
                # Captures how the user starts a new conversation after a long silence.
                if (
                    i == 0
                    or (msg["timestamp"] and consolidated_messages[i - 1]["timestamp"]
                        and msg["timestamp"] - consolidated_messages[i - 1]["timestamp"] > timedelta(hours=4))
                ):
                    all_training_examples.append({
                        "prompt": "[CONVERSATION_START]",
                        "response": msg["content"],
                        "persona": msg["persona"]
                    })

                # Example Type 2: CONTEXT-AWARE RESPONSES
                # Captures how the user responds to recent messages.
                else:
                    context_messages = []
                    # Look at the previous 4 messages for context.
                    start_idx = max(0, i - 4)
                    for j in range(start_idx, i):
                        prev_msg = consolidated_messages[j]
                        # Ignore context that is more than 6 hours old.
                        if (
                            msg["timestamp"]
                            and prev_msg["timestamp"]
                            and msg["timestamp"] - prev_msg["timestamp"] > timedelta(hours=6)
                        ):
                            continue
                        context_messages.append(f"{prev_msg['sender']}: {prev_msg['content']}")

                    if context_messages:
                        prompt = " [TURN_BREAK] ".join(context_messages)
                        all_training_examples.append({
                            "prompt": prompt,
                            "response": msg["content"],
                            "persona": msg["persona"]
                        })

                # Example Type 3: DIRECT Q&A PAIRS
                # Captures how the user answers direct questions.
                if i > 0:
                    prev_msg = consolidated_messages[i - 1]
                    if prev_msg["sender"] != user_sender_name:
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
                                "persona": msg["persona"]
                            })

                # Example Type 4: TOPIC TRANSITIONS
                # Captures how the user changes the subject.
                if (
                    msg["sender"] == user_sender_name
                    and i > 1
                    and consolidated_messages[i - 1]["sender"] != user_sender_name
                ):
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
                            "persona": msg["persona"]
                        })

    # 4. Write all generated examples to the output CSV file.
    with open(output_csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response", "persona"])
        for example in all_training_examples:
            writer.writerow([
                example["prompt"],
                example["response"],
                example["persona"]
            ])

    # 5. Print statistics about the generated dataset.
    total_examples = len(all_training_examples)
    print(f"Successfully created enhanced dataset: {output_csv_file}")
    print(f"Total training examples: {total_examples}")


def create_vector_database(csv_path, text_column, db_path, model_name):
    """
    Creates a FAISS vector database from the prompts in the generated CSV file.
    """
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
    """
    Validates that the provided sender name exists in the chat files.
    Checks the first few files and lines for efficiency.
    """
    line_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s[AP]M) - ([^:]+): (.*)")
    found_senders = set()

    print("Validating sender name...")
    # Check first 3 files for efficiency.
    for chat_file in chat_files[:3]:
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                # Check first 50 lines of each file.
                for line in f.readlines()[:50]:
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
    """Saves the validated sender name to a file for the chat script to use."""
    sender_file_path = os.path.join(temp_dir, 'sender_name.txt')
    with open(sender_file_path, 'w', encoding='utf-8') as f:
        f.write(sender_name)
    print(f"✓ Sender name saved to {sender_file_path}")


def display_persona_statistics(chat_files):
    """Displays statistics about the personas found in the chat files."""
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

def consolidate_txt_files_in_directory(parent_dir):
    """
    Consolidates multiple .txt files in each subfolder into a single file
    named after the folder.
    """
    if not os.path.exists(parent_dir):
        return

    # Get all subdirectories
    subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

    for folder in subfolders:
        # Look for txt files recursively in case zip extraction created nested folders
        txt_files = glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True)

        if len(txt_files) >= 2:
            folder_name = os.path.basename(folder)
            consolidated_path = os.path.join(folder, f"{folder_name}.txt")

            print(f"Consolidating {len(txt_files)} files in folder '{folder_name}'...")

            # Create consolidated file
            with open(consolidated_path, 'w', encoding='utf-8') as outfile:
                for i, txt_file in enumerate(txt_files):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as infile:
                            content = infile.read().strip()
                            if content:  # Only add non-empty content
                                outfile.write(content)
                                # Add separator between files (except for the last one)
                                if i < len(txt_files) - 1:
                                    outfile.write('\n\n')
                    except Exception as e:
                        print(f"Warning: Could not read {txt_file}: {str(e)}")

            # Remove original files (except the consolidated one)
            files_removed = 0
            for txt_file in txt_files:
                if txt_file != consolidated_path:
                    try:
                        os.remove(txt_file)
                        files_removed += 1
                    except Exception as e:
                        print(f"Warning: Could not remove {txt_file}: {str(e)}")

            print(f"Consolidated into '{folder_name}.txt' and removed {files_removed} original files.")

def sanitize_txt_basename(basename: str) -> str:
    """
    Remove the 'WhatsApp Chat with ' prefix (case-insensitive) from .txt filenames.
    """
    name, ext = os.path.splitext(basename)
    if ext.lower() != ".txt":
        return basename
    new_name = _PREFIX_RE.sub("", name, count=1).strip()
    if not new_name:
        new_name = name
    return f"{new_name}{ext}"

def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")

def extract_all_zip_files(base_dir: str) -> bool:
    """
    Extract all .zip files under base_dir.
    - Only reads .txt members.
    - Strips 'WhatsApp Chat with ' from basenames.
    - Merges same sanitized names ordered by each ZIP member's timestamp.
    - If a sanitized .txt already exists on disk, include it (by mtime) in the merge.
    Returns True if anything was extracted or merged, else False.
    """
    extracted_any = False
    pending_merges = {}  # out_path -> list[(timestamp, text)]

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.lower().endswith(".zip"):
                continue
            zip_path = os.path.join(root, fname)
            try:
                with zipfile.ZipFile(zip_path, mode="r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        base = os.path.basename(info.filename)
                        if not base.lower().endswith(".txt"):
                            # Skip non-.txt from ZIP exports
                            continue

                        # Sanitize the basename and build output path
                        sanitized = sanitize_txt_basename(base)
                        out_path = os.path.join(root, sanitized)

                        with zf.open(info, "r") as src:
                            data = src.read()

                        text = _decode_text(data)
                        ts = datetime(*info.date_time).timestamp()
                        pending_merges.setdefault(out_path, []).append((ts, text))
                        extracted_any = True
            except zipfile.BadZipFile:
                # Skip invalid ZIPs gracefully
                continue

    # Include preexisting sanitized/unsanitized .txt files on disk in merges (by mtime)
    for out_path, items in list(pending_merges.items()):
        # Existing sanitized file
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                items.append((os.path.getmtime(out_path), f.read()))

        # Legacy unsanitized variant (prefix present) to fold in
        legacy = os.path.join(os.path.dirname(out_path), "WhatsApp Chat with " + os.path.basename(out_path))
        if os.path.exists(legacy):
            with open(legacy, "r", encoding="utf-8", errors="ignore") as f:
                items.append((os.path.getmtime(legacy), f.read()))
            try:
                os.remove(legacy)
            except OSError:
                pass

        # Sort older first and write merged file
        items.sort(key=lambda x: x)
        merged = "\n".join(s.strip("\n") for _, s in items) + "\n"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(merged)

    return extracted_any

if __name__ == "__main__":
    print("--- Starting Chatbot Setup ---")

    # Step 1: Check for and create necessary directories.
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    if not os.path.exists(EXTRACTED_CHATS_DIR):
        os.makedirs(os.path.join(EXTRACTED_CHATS_DIR, 'Personal'))
        os.makedirs(os.path.join(EXTRACTED_CHATS_DIR, 'Group'))
        print(f"IMPORTANT: The directory structure has been created in '{EXTRACTED_CHATS_DIR}'.")
        print("Please place your exported chat .txt files or .zip files in the 'Personal' and 'Group' folders.")
        print(
            "You can organize them in subfolders - multiple files in the same subfolder will be automatically consolidated in a single persona.")
        input("\nPress Enter after you have placed your chat files in the folders...")

    # Step 2: Always extract all zip files first (regardless of existing txt files)
    print("\nSearching for and extracting zip files...")
    personal_extraction = extract_all_zip_files(os.path.join(EXTRACTED_CHATS_DIR, 'Personal'))
    group_extraction = extract_all_zip_files(os.path.join(EXTRACTED_CHATS_DIR, 'Group'))

    if personal_extraction or group_extraction:
        print("✓ Zip extraction completed!")
    else:
        print("No zip files found to extract.")

    # Step 3: Find all txt files after extraction
    print("\nScanning for .txt files...")
    personal_chats = glob.glob(os.path.join(EXTRACTED_CHATS_DIR, 'Personal', '**', '*.txt'), recursive=True)
    group_chats = glob.glob(os.path.join(EXTRACTED_CHATS_DIR, 'Group', '**', '*.txt'), recursive=True)
    all_chat_files = personal_chats + group_chats

    if all_chat_files:
        print(f"Found {len(all_chat_files)} .txt file(s) total.")
    else:
        print("No .txt files found.")

    # Step 4: Consolidate multiple txt files in subfolders (AFTER extraction is complete)
    if all_chat_files:
        print("\nChecking for consolidation opportunities...")

        print("Checking Personal directory...")
        consolidate_txt_files_in_directory(os.path.join(EXTRACTED_CHATS_DIR, 'Personal'))

        print("Checking Group directory...")
        consolidate_txt_files_in_directory(os.path.join(EXTRACTED_CHATS_DIR, 'Group'))

        print("✓ Consolidation check completed!")

        print("\nFinal scan after consolidation...")
        personal_chats = glob.glob(os.path.join(EXTRACTED_CHATS_DIR, 'Personal', '**', '*.txt'), recursive=True)
        group_chats = glob.glob(os.path.join(EXTRACTED_CHATS_DIR, 'Group', '**', '*.txt'), recursive=True)
        all_chat_files = personal_chats + group_chats

        print(f"Final count: {len(all_chat_files)} chat file(s) ready for processing.")

    # Step 5: Final check for chat files
    if not all_chat_files:
        print("WARNING: No chat files found. Please add .txt or .zip files to the 'extracted_chats' directory.")
        sys.exit(0)

    display_persona_statistics(all_chat_files)

    # Step 3: Get and validate the user's sender name.
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

    # Step 4: Generate the dataset and create the vector database.
    create_context_dataset(all_chat_files, OUTPUT_CSV_PATH, sender_name)
    create_vector_database(OUTPUT_CSV_PATH, TEXT_COLUMN, DB_PATH, MODEL_NAME)

    print("\n--- Setup Complete ---")
    print("You can now run the chatbot using: python gemini_chat.py")
