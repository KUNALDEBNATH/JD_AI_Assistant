# JD AI Assistant

A conversational AI assistant built with an ollama model(llama 3.2) with a sleek, modern web interface. JD features persistent conversation memory, real-time text-to-speech capabilities, automatically logs conversations in JSONL format for model fine-tuning in real-time and an intuitive multi-conversation management system.

![JD Assistant](https://img.shields.io/badge/AI-Assistant-purple?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-%23000000?style=for-the-badge&logo=ollama&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)

## Features

- **🤖 Llama 3.2 Integration**: Llama 3.2 model is integrated via Ollama
- **💾 Persistent Memory**: All conversations are saved and can be resumed later
- **🎙️ Text-to-Speech**: Optional voice output for AI responses using gTTS
- **📚 Multi-Conversation Management**: Switch between different conversation topics seamlessly
- **🎨 Modern UI**: Beautiful, animated dark-themed interface with smooth transitions
- **📝 Training Data Export**: Automatically logs conversations in JSONL format for model fine-tuning
- **✏️ Edit & Copy Tools**: Quickly edit your last prompt or copy AI responses
- **🔄 Context-Aware**: Maintains context across conversations using memory of past interactions

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab (recommended) or Linux environment
- Google Drive (for persistent storage)

### Setup Instructions

1. **Clone or upload the script** to your Google Colab notebook or local environment.

2. **Install dependencies**:
   The script automatically installs required packages:
   - `ollama` - For running Llama models locally
   - `gtts` - Google Text-to-Speech
   - `gradio` - Web UI framework

3. **Configure storage path**:
   Edit the `BASE_DIR` variable in the script to point to your desired Google Drive folder:
   ```python
   BASE_DIR = "/content/drive/MyDrive/JD"  # Change this to your folder
   ```

4. **Run the script**:
   The installation process will:
   - Install Ollama
   - Download the Llama 3.2 model
   - Start the Ollama server
   - Launch the Gradio interface

## Usage

### Starting JD

Run all cells in the notebook. The final cell will launch the Gradio interface and provide a shareable link.

### Interface Overview

**Sidebar (Left)**
- **New Chat Button**: Start a fresh conversation
- **Conversations List**: View and switch between saved conversation topics
- Conversations are automatically titled using the first few words of your initial message

**Main Chat Area (Right)**
- **Chat Display**: View your conversation history with JD
- **Input Box**: Type your messages here
- **Send Button**: Submit your message
- **Tools**:
  - ✎ Edit last prompt - Reload your previous message for editing
  - ⧉ Copy last answer - Copy JD's last response to clipboard
  - JD server voice - Toggle text-to-speech on/off

### Managing Conversations

- **Create New Chat**: Click "＋ New chat" to start a fresh topic
- **Switch Conversations**: Select any conversation from the sidebar to view its history
- **Auto-Save**: All conversations are automatically saved to your Google Drive
- **Memory System**: JD remembers context from up to 50 previous conversation turns across all topics

## Data Storage

JD maintains two types of data files in your specified directory:

### 1. `conversations.json`
Structured storage of all conversation topics:
```json
[
  {
    "id": "unique-uuid",
    "title": "First few words of conversation...",
    "turns": [
      ["user message", "assistant response"],
      ...
    ]
  }
]
```

### 2. `train.jsonl`
Flat log format suitable for model training:
```json
{"instruction": "user message", "output": "assistant response"}
{"instruction": "another message", "output": "another response"}
```

### 3. `audio/` directory
Temporary storage for generated TTS audio files.

## Customization

### Change the AI Model

Modify the `MODEL_NAME` variable to use different Ollama models:
```python
MODEL_NAME = "llama3.2"  # Change to "llama2", "mistral", etc.
```

Remember to pull the new model first:
```bash
!ollama pull your-model-name
```

### Customize the System Prompt

Edit the system message in `build_messages_from_memory()`:
```python
system_msg = {
    "role": "system",
    "content": (
        "You are JD, an AI Assistant created by Kunal Debnath Sir. "
        "Customize this message to change JD's personality and behavior."
    ),
}
```

### Adjust Memory Length

Change how many past conversation turns are included in context:
```python
for p in mem_pairs[-50:]:  # Change 50 to your desired number
```

### Modify UI Theme

The custom CSS in `CUSTOM_CSS` can be edited to change colors, animations, and layout. Key color variables:
- Background: `#050508` (dark grey)
- Accent colors: `#a855f7` (purple), `#f97316` (orange)
- Text: `#e5e7eb` (light grey)

## Technical Architecture

### Components

1. **Ollama Server**: Runs locally to serve the Llama 3.2 model
2. **Memory System**: Loads all past conversations to provide context
3. **Gradio Interface**: Provides the web-based chat UI
4. **Storage Layer**: Manages JSON and JSONL file operations
5. **TTS Engine**: Converts text responses to speech

### Workflow

1. User sends a message
2. System loads relevant conversation context (up to 50 past turns)
3. Message is sent to Llama 3.2 via Ollama
4. Response is displayed and optionally converted to speech
5. Conversation is saved to both JSON and JSONL formats
6. UI updates with new conversation state

## Troubleshooting

### Ollama Server Issues

If the chat isn't responding:
```bash
# Check if Ollama is running
!ps aux | grep ollama

# Restart Ollama server
!pkill -f ollama
!nohup ollama serve > ollama.log 2>&1 &
```

### Model Not Found

Pull the model again:
```bash
!ollama pull llama3.2
```

### Storage Errors

Ensure your Google Drive is mounted:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Memory Issues

If conversations become too large, reduce the memory window:
```python
for p in mem_pairs[-20]:  # Reduced from 50
```

## Performance Notes

- **First Response**: May take 10-20 seconds as the model loads
- **Subsequent Responses**: Typically 2-5 seconds depending on complexity
- **Memory Usage**: Increases with conversation length; monitor RAM in Colab
- **Storage**: Each conversation typically uses a few KB of storage

## Credits

- **Created by**: Kunal Debnath
- **AI Model**: Llama 3.2 
- **Framework**: Gradio for UI, Ollama for model serving
- **TTS**: Google Text-to-Speech (gTTS)

## Contributing

Feel free to fork this project and customize it for your needs. Suggestions for improvements:
- Add conversation search functionality
- Implement conversation export/import
- Add support for image inputs
- Create conversation summarization
- Add multi-language support

**Note**: This assistant is designed to run in Google Colab with GPU support for optimal performance. CPU-only execution is possible but will be significantly slower.
