# Mythryl

This project aims to create an easy-to-use script for building a RAG-based personalized chatbot using social interaction data (currently limited to WhatsApp chats).

## Features

- **Intelligent Conversation Context Analysis:** The bot creates diverse training datasets with conversation initiations, contextual responses, direct Q&As, and topic transitions.
- **RAG-Powered Pipeline:** Implements Retrieval-Augmented Generation (RAG) using FAISS vector search to find relevant conversation examples and enhance AI responses with authentic communication patterns.
- **Context-Aware Response Generation:** Combines vector similarity search and actual conversation history to generate replies that reflect both the style and context of your chosen persona.
- **One-Click Setup & Auto-Configuration:** Automatically processes WhatsApp chat exports, creates vector databases, validates sender names, and sets up the entire RAG pipeline with minimal intervention from you!!

## How It Works

### **Setup Phase: `setup.py`**

**Initial Setup & Validation:**
- Creates temp/ and extracted_chats/ directories, each with Personal/ and Group/ subfolders, then waits for you to upload chat files
- Scans for WhatsApp chat `.txt` files and displays persona statistics
- Validates the if the provided sender name exists in the chat files before proceeding
- Saves sender name to `temp/sender_name.txt` for future use

**Intelligent Chat Processing:**
- Parses WhatsApp export format using regex (timestamp, sender, message)
- Consolidates consecutive messages from the same sender and detects conversation breaks (gaps of several hours)
- Creates 4 types of training examples: conversation starters, contextual responses, direct Q&A pairs, and topic transitions
- Exports processed data to `temp/persona_style_v2.csv` with relevant metadata

**Vector Database Creation:**
- Encodes all conversation prompts using SentenceTransformer model
- Builds FAISS vector index for semantic similarity search
- Saves index to `temp/style_v2.index` for fast retrieval


### **Chat Phase: `gemini_chat.py`**

**RAG-Powered Response Generation:**
- Loads FAISS index, conversation dataset, and AI model
- Performs semantic search to find similar conversation examples for your query
- Combines retrieved examples with conversation history to create context-rich prompts
- Uses Google Gemini to generate responses that match the persona’s communication style

## Setup and Usage

### 1. Installation  

Use Python 3.9 – 3.11. (Optional but recommended) Set up a virtual environment:
 ```bash
  python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
  ```
Then install the requirements:  
  ```bash
  pip install -r requirements.txt
  ```

### 2. Data Generation  

To get the chatbot ready, run the setup script and just follow the on-screen instructions:
   ```bash
   python setup.py
   ```
**Important:** 
If not running for the first time place each exported WhatsApp .txt file in the auto created folder `extracted_chats` before running the setup script

### 3. Setting an API key  

Save your API key in the following format in a `.env` file located in the same directory as your `gemini_chat.py` file:
```txt
API_KEY=YOUR_GEMINI_API_KEY_HERE
```

### 4. Running the Chatbot 

Once your data files are ready, you can chat with your personalized AI by running:
```bash
python gemini_chat.py
```
Inside the chat session: type **`switch`** to change persona or **`quit`** to exit.

## Config

Currently, the only configuration required is adding your API key to the .env file.
If you want to enable DEBUG mode or change specific paths or models, you can do that directly in the scripts.

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

## Contact

Feel free to reach out if you have questions, suggestions, or want to collaborate!
Email: animesh_varma@protonmail.com

**NOTE:** I’m a high school student building this project in any spare time I can find, so contributors and general advice are always more than welcomed!
Also, this is my first serious project, so please excuse any small mistakes :-) 
