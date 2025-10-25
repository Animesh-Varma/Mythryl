# core.py

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import google.generativeai as genai
import ollama
import re
import numpy as np
from thefuzz import process

# --- Global Settings ---
load_dotenv()
DEBUG = os.getenv("DEBUG", False)
INDEX_PATH = "temp/style_v2.index"
CSV_PATH = "temp/persona_style_v2.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# --- LLM Parameters ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.9))
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", 6))
VECTOR_DB_SEARCH_COUNT = int(os.getenv("VECTOR_DB_SEARCH_COUNT", 5))

def log_debug(message):
    """Prints a debug message to the console if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {message}")


def load_resources(index_path=INDEX_PATH, csv_path=CSV_PATH, model_name=MODEL_NAME):
    """Loads the FAISS index, persona CSV, and sentence transformer model."""
    print("Loading style databases...", end="", flush=True)
    try:
        if not os.path.exists(index_path) or not os.path.exists(csv_path):
            print(f"\nError: Database files not found.")
            print("Please run the setup script first: python setup.py")
            return None, None, None

        index = faiss.read_index(index_path)
        df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        model = SentenceTransformer(model_name)
        print("Done.")
        return index, df, model
    except Exception as e:
        print(f"\nAn unexpected error occurred while loading resources: {e}")
        return None, None, None

def find_similar_responses(query, index, df, model, persona, k):
    """Finds k most similar responses from the database for the given persona."""
    if persona == "Generic":
        return pd.DataFrame()  # Return empty DataFrame for generic responses.

    query_vector = model.encode([query])
    # Search for more candidates initially to ensure we find enough for the persona.
    distances, indices = index.search(query_vector, k * 10)

    valid_indices = [i for i in indices[0] if i < len(df)]
    retrieved_df = df.iloc[valid_indices]
    persona_specific_df = retrieved_df[retrieved_df['persona'] == persona]

    # If no direct matches, fall back to the persona's general examples.
    if persona_specific_df.empty:
        similar_df = df[df['persona'] == persona].head(k)
        log_debug(f"Found {len(similar_df)} similar responses (fallback)")
        return similar_df

    if not persona_specific_df.empty:
        log_debug(f"=== EXTRACTED DATABASE VALUES for '{persona}' ===")
        for idx, (_, row) in enumerate(persona_specific_df.head(k).iterrows()):
            log_debug(f"  Prompt: {row['prompt'][:100]}...")
            log_debug(f"  Response: {row['response'][:100]}...")
        log_debug("=== END EXTRACTED VALUES ===")

    log_debug(f"Found {len(persona_specific_df)} similar responses")
    return persona_specific_df.head(k)


def generate_llm_prompt(query, conversation_history, similar_responses, persona, sender_name):
    """Constructs the final prompt for the LLM based on context and persona."""
    style_examples = ""
    if not similar_responses.empty:
        for _, row in similar_responses.iterrows():
            style_examples += f"Context/Prompt: \"{row['prompt']}\"\n"
            style_examples += f"{persona}'s Response: \"{row['response']}\"\n\n"

    # Use the last N messages to keep the context relevant.
    history = "\n".join(conversation_history[-LLM_CONTEXT_SIZE:])

    if persona == "Generic":
        persona_instructions = f"You are {sender_name}. Respond naturally as yourself."
    else:
        persona_instructions = f"""
    You are {sender_name} responding in a '{persona}' context. 
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

def get_llm_response(service_choice, prompt_for_llm, ollama_model_name=None):
    """Gets the response from the selected LLM service."""
    bot_response_text = ""
    if service_choice == 'gemini':
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API Key not found. Please set it in your .env file.")
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel(GEMINI_MODEL)
        generation_config = genai.types.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P
        )
        response = llm_model.generate_content(
            prompt_for_llm,
            generation_config=generation_config
        )
        bot_response_text = response.text
    elif service_choice == 'ollama':
        if not ollama_model_name:
            ollama_model_name = os.getenv("OLLAMA_MODEL")
        if not ollama_model_name:
            raise ValueError("Model name not found. Please set it in your .env file.")
        
        response = ollama.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': prompt_for_llm}],
            options={
                'temperature': LLM_TEMPERATURE,
                'top_p': LLM_TOP_P
            }
        )
        bot_response_text = response['message']['content']
        if not DEBUG:
            bot_response_text = re.sub(r'<think>.*?</think>', '', bot_response_text, flags=re.DOTALL).strip()
    else:
        raise ValueError("Invalid service choice.")
        
    return bot_response_text

def add_message_to_database(df, index, model, persona, prompt, response):
    """Adds a new message to the database and updates the index."""
    # 1. Append to CSV
    new_entry = pd.DataFrame([{'prompt': prompt, 'response': response, 'persona': persona}])
    new_entry.to_csv(CSV_PATH, mode='a', header=False, index=False)

    # 2. Update DataFrame
    df = pd.concat([df, new_entry], ignore_index=True)

    # 3. Encode the new prompt
    new_embedding = model.encode([prompt])

    # 4. Add to FAISS index
    index.add(np.array(new_embedding, dtype=np.float32))

    # 5. Save the updated index
    faiss.write_index(index, INDEX_PATH)
    
    log_debug(f"Added new message for persona '{persona}' and updated index.")
    return df, index

def verify_persona_name(df, persona_name):
    """Verifies a persona name, suggesting the closest match if not found."""
    available_personas = df['persona'].unique().tolist()
    if persona_name in available_personas:
        return {"status": "exact_match", "persona": persona_name}
    else:
        best_match, score = process.extractOne(persona_name, available_personas)
        if score > 80:  # Confidence threshold
            return {"status": "closest_match", "persona": best_match, "confidence": score}
        else:
            return {"status": "not_found"}

