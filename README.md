# Mythryl

This project aims to create an easy-to-use script for building a RAG-based personalized chatbot using social interaction data (currently limited to WhatsApp chats).

## Features

- **Multi-Service Support:** Works with both cloud-based (Gemini) and local (Ollama) language models.
- **Automatic Data Processing:** Extracts chat logs from `.zip` files directly from whatsapp
- **Intelligent Conversation Context Analysis:** The bot creates diverse training datasets with conversation initiations, contextual responses, direct Q&As, and topic transitions.
- **RAG-Powered Pipeline:** Implements Retrieval-Augmented Generation (RAG) using FAISS vector search to find relevant conversation examples and enhance AI responses with authentic communication patterns.
- **Context-Aware Response Generation:** Combines vector similarity search and actual conversation history to generate replies that reflect both the style and context of your chosen persona.
- **Configurable LLM Parameters:** Easily customize LLM temperature, top-p, context size, and search count via the env variables 
- **One-Click Setup & Auto-Configuration:** Automatically processes WhatsApp chat exports, creates vector databases, validates sender names, and sets up the entire RAG pipeline with minimal intervention from you!!

## How It Works

### **Setup Phase: `setup.py`**

**Initial Setup & Validation:**
- Creates `temp/` and `extracted_chats/` directories, each with `Personal/` and `Group/` subfolders, then waits for you to upload chat files.
- **Automatically extracts `.zip` archives and consolidates `.txt` files**, simplifying persona management.
- Scans for WhatsApp chat `.txt` files and displays persona statistics.
- Validates that the provided sender name exists in the chat files before proceeding.
- Saves sender name to `temp/sender_name.txt` for future use.

**Chat Processing:**
- Parses WhatsApp export format using regex (timestamp, sender, message).
- Consolidates consecutive messages from the same sender and detects conversation breaks (gaps of several hours).
- Creates 4 types of training examples: **conversation starters, contextual responses, direct Q&A pairs, and topic transitions.**
- Exports processed data to `temp/persona_style_v2.csv` with relevant metadata.

**Vector Database Creation:**
- Encodes all conversation prompts using the `all-MiniLM-L6-v2` SentenceTransformer model.
- Builds a FAISS vector index for semantic similarity search.
- Saves the index to `temp/style_v2.index` for fast retrieval.


### **Chat Phase: `chat.py`**

**Service Selection:**
- Prompts you to choose between the `gemini` (cloud) or `ollama` (local) service.

**RAG-Powered Response Generation:**
- Loads the FAISS index, conversation dataset, and the selected AI model.
- Performs a semantic search to find similar conversation examples for your query.
- Combines retrieved examples with conversation history to create context-rich prompts for th LLM.
- Uses the chosen LLM (Gemini or Ollama) to generate responses that match the communication style for the given persona.

## Setup and Usage

### 1. Installation  

Clone and enter the repo:
 ```bash
  git clone https://github.com/Animesh-Varma/Mythryl.git
  cd Mythryl
  ```

Use Python 3.9 – 3.11. (Optional but recommended) Set up a virtual environment:
 ```bash
  python -m venv .venv 
  ```
Activate your virtual environment:
 ```bash
  source .venv/bin/activate  # on Windows: .venv\Scripts\activate
  ```

Then install the requirements:

**Note**: You can customize the installation based on your needs. By default, it will install dependencies for both online and offline inference.
- For online inference only: Remove `ollama` from `requirements.txt`.
- For offline inference only: Remove `google-generativeai` from `requirements.txt`.

  ```bash
  pip install -r requirements.txt
  ```

### 2. Data Generation  

To get the chatbot ready, run the setup script and just follow the on-screen instructions:
   ```bash
   python setup.py
   ```
**Important:** 
If not running for the first time place each exported WhatsApp .txt/.zip file in the auto created folder `extracted_chats` before running the setup script

### 3. Setting Environment Variables

Create a `.env` file in the root directory and add the following variables.

- **For Gemini (Cloud-based):**
  ```env
  API_KEY=YOUR_GEMINI_API_KEY_HERE
  ```

- **For Ollama (Local):**
  ```env
  OLLAMA_MODEL=NAME_OF_THE_LOCAL_MODEL_TO_USE
  ```

- **Optional LLM Parameters:**
  ```env
  LLM_TEMPERATURE=0.7 
  LLM_TOP_P=0.9
  LLM_CONTEXT_SIZE=6
  VECTOR_DB_SEARCH_COUNT=5
  ```
- **Optional Script Parameters:**
  ```env
  DEBUG = False
  ```
  
### 4. Running the Chatbot 

Once your data files are ready, you can chat with your personalized AI by running:
```bash
python chat.py
```
Inside the chat session: type **`switch`** to change persona or **`quit`** to exit.

## API Usage

Mythryl includes a local API server, allowing you to integrate your personalized chatbot with other applications.
To use the API first run and follow the `setup.py`, then start the server:

```bash
python api.py
```

Then the API will be available at `http://127.0.0.1:50507`.

### Endpoints

#### GET /personas

Returns a list of all available personas.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:50507/personas"
```

**Example Response:**
```json
{
  "personas": [
    "persona1",
    "persona2",
    "persona3"
  ]
}
```

#### POST /verify_persona

Verifies if the persona is available. If an exact match is not found, it suggests the closest match.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:50507/verify_persona" -H "Content-Type: application/json" -d '{
  "persona": "persna1"
}'
```

**Example Response (Closest Match):**
```json
{
  "status": "closest_match",
  "persona": "persona1",
  "confidence": 86
}
```

#### POST /chat

Handles a chat request with a specific persona.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:50507/chat" -H "Content-Type: application/json" -d '{
  "persona": "persona1",
  "message": "Hello, how are you?",
  "service": "gemini"
}'
```

**Example Response:**
```json
{
  "response": "I am doing well, thank you!"
}
```

#### POST /add_message

Adds a new message to the vector database for the specified persona.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:50507/add_message" -H "Content-Type: application/json" -d '{
  "persona": "persona1",
  "prompt": "This is a new prompt.",
  "response": "This is a new response."
}'
```

**Example Response:**
```json
{
  "message": "Message added successfully."
}
```

## Config

Configuration is managed through the `.env` file. You can set your API key, choose your Ollama model, and adjust LLM and script parameters. 
If you want to change specific paths, gemini models or system prompts, you can do that directly in the scripts.
[I'd suggest using gemini for generation as it works a lot better in my testing, the default gemini model is 2.5 flash]

## TODO & Contributions

Contribute if you can, issues and feature requests are always appreciated!
Future planned features, tasks, and any pending fixes are listed in the [TODO.md](TODO.md) (Some items are a bit vague, so feel free to email me if you need clarification.)
Feel free to open a pull request or issue for new features, bug fixes, or suggestions. Everyone is more than welcome!

## Tested on

1. Windows Home Version 24H2
2. Arch Linux 
3. Debian 6.1.135-1 (aarch64, running on a terminal emulator on my Pixel 9)

## Use of AI

The use of AI in development is now inevitable,trying to avoid it is simply impractical.
In my humble view, the best approach is to use AI for bulk raw generation, and then fine-tune the results manually.
That’s exactly the philosophy behind this project!
The concept and core implementation are entirely my own, with invaluable assistance from AI systems, especially for rapid prototyping and improving code readability.
Rest assured, all code was thoroughly tested and carefully reviewed by me before release.

## Privacy

This project uses a hybrid approach to data privacy, combining local processing with cloud-based AI services. Here’s how your data is handled at each stage:

### Local Data Processing (`setup.py`)

- **All your chat data stays on your local machine during setup.** The `setup.py` script reads your chat export files from the `extracted_chats` directory and processes them locally.
- Generated files (`persona_style_v2.csv`, `style_v2.index`, and `sender_name.txt`) are stored in the `temp` directory on your computer.
- **No chat data is sent to any external server or cloud service during this phase.** The sentence transformer model for vector embeddings is downloaded and runs entirely on your machine.

### Cloud-Based AI Interaction (`chat.py`)

*// only applicable if using gemini as the provider, choosing ollama instead does all the processing locally*

- When you chat with the bot, certain pieces of information are sent to the **Google Gemini API** to generate responses. This is the only time your data leaves your local device.
- The data sent to Gemini API includes:
    - The message you type (your query)
    - The last 6 messages of the ongoing conversation for context
    - A few relevant examples from your own chat history (retrieved from your local `persona_style_v2.csv`) to help the AI match your style
    - Your sender name and chosen persona context
- Your **API key**, stored in the `.env` file, is used for secure authentication with the Gemini API.

## Contact

Feel free to reach out if you have questions, suggestions, or want to collaborate!
Email: `animesh_varma@protonmail.com`

**NOTE:** I’m a high school student building this project in any spare time I can find, so contributors and general advice are always more than welcomed!
Also, this is my first serious project, so please excuse any small mistakes :-) 
