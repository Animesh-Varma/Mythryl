import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import sys
import io
import time
import warnings
import os
from dotenv import load_dotenv
import threading

DEBUG = False

def log_debug(message):
    if DEBUG:
        print(f"[DEBUG] {message}")

# Suppress the specific FutureWarning from the transformers library
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# --- Force UTF-8 for all output ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

# --- Configuration ---
INDEX_PATH = "temp/style_v2.index"
CSV_PATH = "temp/persona_style_v2.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"


def load_resources(index_path, csv_path, model_name):
    print("Loading style databases...", end="", flush=True)
    try:
        if not os.path.exists(index_path):
            print(f"\nError: Index file not found at {index_path}")
            print("Please run the setup script first: python setup.py")
            return None, None, None
        if not os.path.exists(csv_path):
            print(f"\nError: CSV file not found at {csv_path}")
            print("Please run the setup script first: python setup.py")
            return None, None, None

        index = faiss.read_index(index_path)
        df = pd.read_csv(csv_path).dropna()

        # Handle both old and new CSV formats
        if 'type' not in df.columns:
            df['type'] = 'legacy'

        model = SentenceTransformer(model_name)
        print("Done.")
        return index, df, model
    except Exception as e:
        print(f"\nAn unexpected error occurred while loading resources: {e}")
        return None, None, None

def get_persona_choice(df):
    personas = df['persona'].unique().tolist()

    print("\nAvailable personas for impersonation:")
    print("=" * 50)

    # Show persona stats
    for i, persona in enumerate(personas):
        persona_data = df[df['persona'] == persona]
        total_examples = len(persona_data)

        if 'type' in df.columns:
            type_counts = persona_data['type'].value_counts()
            type_info = ", ".join([f"{t.replace('_', ' ')}: {c}" for t, c in type_counts.items()])
            print(f"[{i + 1}] {persona} ({total_examples} examples)")
            print(f"    Training types: {type_info}")
        else:
            print(f"[{i + 1}] {persona} ({total_examples} examples)")

    print(f"[{len(personas) + 1}] Generic Response (no persona)")
    print("=" * 50)

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(personas):
                chosen_persona = personas[choice - 1]
                log_debug(f"Persona selected: {chosen_persona}")
                return chosen_persona
            elif choice == len(personas) + 1:
                log_debug("Generic response selected")
                return "Generic"
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def find_similar_responses(query, index, df, model, persona, k=5):
    if persona == "Generic":
        return pd.DataFrame()  # No style examples for generic

    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k * 10)  # Get more candidates

    valid_indices = [i for i in indices[0] if i < len(df)]
    retrieved_df = df.iloc[valid_indices]
    persona_specific_df = retrieved_df[retrieved_df['persona'] == persona]

    if persona_specific_df.empty:
        similar_df = df[df['persona'] == persona].head(k)
        log_debug(f"Found {len(similar_df)} similar responses (fallback)")
        return similar_df

    # Prioritize diverse response types if available
    if 'type' in persona_specific_df.columns:
        diverse_responses = []
        types_seen = set()

        for _, row in persona_specific_df.iterrows():
            if len(diverse_responses) >= k:
                break

            response_type = row.get('type', 'unknown')
            if response_type not in types_seen or len(types_seen) >= 3:
                diverse_responses.append(row)
                types_seen.add(response_type)

        result_df = pd.DataFrame(diverse_responses)
        log_debug(f"Found {len(result_df)} diverse responses with types: {types_seen}")
        return result_df

    if not persona_specific_df.empty:
        log_debug(f"=== EXTRACTED DATABASE VALUES for '{persona}' ===")
        for idx, (_, row) in enumerate(persona_specific_df.head(k).iterrows()):
            response_type = row.get('type', 'unknown')
            log_debug(f"Example {idx + 1} [{response_type}]:")
            log_debug(f"  Prompt: {row['prompt'][:100]}...")
            log_debug(f"  Response: {row['response'][:100]}...")
        log_debug("=== END EXTRACTED VALUES ===")

    log_debug(f"Found {len(persona_specific_df)} similar responses")
    return persona_specific_df.head(k)

def generate_gemini_prompt(query, conversation_history, similar_responses, persona, sender_name):
    style_examples = ""
    if not similar_responses.empty:
        for _, row in similar_responses.iterrows():
            response_type = row.get('type', 'response')
            type_label = response_type.replace('_', ' ').title()

            style_examples += f"[{type_label}]\n"
            style_examples += f"Context/Prompt: \"{row['prompt']}\"\n"
            style_examples += f"{persona}'s Response: \"{row['response']}\"\n\n"

    history = "\n".join(conversation_history[-6:])  # Limit history

    if persona == "Generic":
        persona_instructions = f"You are {sender_name}. Respond naturally as yourself."
    else:
        persona_instructions = f"""You are {sender_name} responding in a '{persona}' context. 
    Study how {sender_name} communicates in {persona} situations from the examples and respond exactly as {sender_name} would in this context."""

    prompt = f"""
    **CRITICAL INSTRUCTION:** You are {sender_name}. Respond as {sender_name} would in a '{persona}' context.

    **CONTEXT-AWARE RESPONSE PROTOCOL:**
    1. **ANALYZE SILENTLY:** Study how {sender_name} communicates in '{persona}' situations
    2. **EMBODY THE CONTEXT:** Channel {sender_name}'s tone/style for {persona} interactions
    3. **RESPOND AS {sender_name.upper()}:** Match {sender_name}'s authentic communication patterns for this context

    --- Recent Conversation ---
    {history}
    ---------------------------

    --- {sender_name}'s Communication Patterns in '{persona}' Context ---
    {style_examples}
    --- End of {sender_name}'s {persona} Patterns ---

    **Current Message to Respond To:**
    User: "{query}"

    **Your response as {sender_name} in {persona} context:**"""

    log_debug(f"Generated prompt for persona '{persona}'")
    return prompt

def get_multi_line_input(persona):
    if persona == "Generic":
        print("You (end with an empty line):")
    else:
        print(f"{persona} (end with an empty line):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return " [MSG_BREAK] ".join(lines)

def spinning_animation(stop_event):
    """Displays a spinning animation until the stop_event is set."""
    animation = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rBot is thinking... {animation[idx % len(animation)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 30 + '\r') # Clear the line
    sys.stdout.flush()

def main():
    try:
        api_key = os.getenv("API_KEY")
        if not api_key:
            print("API Key not found. Please set it in your .env file.")
            return
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print(f"API Key Error: {e}")
        return

    index, df, model = load_resources(INDEX_PATH, CSV_PATH, MODEL_NAME)
    if index is None: return

    SENDER_NAME_PATH = "temp/sender_name.txt"
    if os.path.exists(SENDER_NAME_PATH):
        with open(SENDER_NAME_PATH, "r", encoding="utf-8") as f:
            sender_name = f.read().strip()
        if not sender_name:
            print(f"Error: 'sender_name.txt' is empty.")
            return
    else:
        sender_name = input("Enter your name (as it appears in chats): ").strip()
        if not sender_name:
            print("Sender name cannot be empty.")
            return

    chosen_persona = get_persona_choice(df)
    conversation_history = []

    # Updated CLI display
    print(f"\n👤 Responding as: {sender_name}")
    if chosen_persona == "Generic":
        print(f"📝 Using: Generic response style")
    else:
        print(f"📝 Using communication style from: {chosen_persona} context")

    print("Commands: 'quit' to exit, 'switch' to change context")
    print("=" * 60)

    while True:
        try:
            user_query = get_multi_line_input(chosen_persona)
            if user_query.lower() == 'quit': break
            if user_query.lower() == 'switch':
                chosen_persona = get_persona_choice(df)
                conversation_history = []

                print(f"\n👤 Still responding as: {sender_name}")
                if chosen_persona == "Generic":
                    print(f"📝 Switched to: Generic response style")
                else:
                    print(f"📝 Switched to: {chosen_persona} context")
                print("=" * 60)
                continue

            similar_responses = find_similar_responses(user_query, index, df, model, chosen_persona)
            prompt_for_gemini = generate_gemini_prompt(user_query, conversation_history, similar_responses, chosen_persona, sender_name)

            log_debug("=== FULL PROMPT BEING SENT TO LLM ===")
            log_debug(prompt_for_gemini)
            log_debug("=== END PROMPT ===")

            stop_animation = threading.Event()
            animation_thread = threading.Thread(target=spinning_animation, args=(stop_animation,))
            animation_thread.start()

            try:
                response = gemini_model.generate_content(prompt_for_gemini)
            finally:
                stop_animation.set()
                animation_thread.join()

            bot_responses = response.text.strip().split("[MSG_BREAK]")

            for i, msg in enumerate(bot_responses):
                msg = msg.strip()
                if msg:
                    print(f"{sender_name}: {msg}")

            if chosen_persona == "Generic":
                conversation_history.append(f"You: {user_query}")
            else:
                conversation_history.append(f"{chosen_persona}: {user_query}")
            conversation_history.append(f"{sender_name}: {response.text.strip()}")
            if len(conversation_history) > 6:
                conversation_history = conversation_history[-6:]

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

    print("\n" + "=" * 60 + "\nChat ended. Goodbye!")

if __name__ == "__main__":
    main()
