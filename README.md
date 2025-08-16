
# Personalized AI Chatbot

This project is a sophisticated, persona-aware AI chatbot that learns your unique conversational style from your WhatsApp chat history. It uses a Retrieval-Augmented Generation (RAG) architecture with the Gemini 1.5 Flash model to conduct natural, coherent, and stylistically accurate conversations.

## Features

- **Persona-Aware:** The bot can adapt its style to different people (e.g., Male Friend, Female Friend, Stranger).
- **Multi-Message Context:** Understands and generates fragmented, multi-message responses to mimic real texting patterns.
- **Conversational Memory:** Remembers the last few turns of the conversation to provide coherent replies.
- **Secure:** Prompts for your API key at runtime and does not store it.

## How It Works

1.  **Data Processing:** The `create_persona_dataset_v2.py` script reads your exported WhatsApp chat logs, identifies different personas, and groups consecutive messages to understand multi-message turns.
2.  **Vector Database:** The `create_vector_db.py` script creates a FAISS vector index from your conversational data, allowing for rapid, semantic searching of your chat history.
3.  **Intelligent Chat:** The `gemini_chat_final.py` script uses this vector database to find relevant examples of your style and feeds them, along with the recent conversation history, to the Gemini model to generate a response in your unique voice.

## Setup and Usage

### 1. Installation

First, you need to install the required Python libraries. Open a terminal in this directory and run:

```bash
pip install -r requirements.txt
```

### 2. Data Generation

To prepare the chatbot, run the setup script. This will create the necessary folders and process your chat data.

**IMPORTANT:** Before running, make sure you have placed your exported WhatsApp `.txt` chat files inside the `extracted_chats` directory.

```bash
python setup.py
```

### 3. Running the Chatbot

Once the data files are generated, you can start a conversation with your personalized AI:

```bash
python gemini_chat_final.py
```

The script will first prompt you to enter your Google AI API key. Then, you will be asked to select a persona to chat as.
