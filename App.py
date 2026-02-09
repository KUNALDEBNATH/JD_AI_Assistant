#!/usr/bin/env python3
"""
JD AI Assistant - Main Application
A conversational AI assistant powered by Llama 3.2 with persistent memory
"""

import os
import json
import uuid
import time
import subprocess
from typing import Optional, List, Tuple

import ollama
from gtts import gTTS
import gradio as gr

# =============== CONFIGURATION ===============
MODEL_NAME = "llama3.2"
BASE_DIR = os.path.join(os.path.expanduser("~"), "jd_data")  # Local storage
os.makedirs(BASE_DIR, exist_ok=True)

CONV_PATH = os.path.join(BASE_DIR, "conversations.json")  # Per-topic chats
FLAT_PATH = os.path.join(BASE_DIR, "train.jsonl")         # Flat log for training

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# =============== DATA MANAGEMENT ===============

def load_conversations(path: str = CONV_PATH) -> List[dict]:
    """Load all conversations from JSON file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"Error loading conversations: {e}")
    return []


def save_conversations(convs: List[dict], path: str = CONV_PATH) -> None:
    """Save all conversations to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)


def log_flat_pair(user_text: str, assistant_text: str, path: str = FLAT_PATH) -> None:
    """Log a single conversation turn to JSONL file for training."""
    if not user_text or not assistant_text:
        return
    rec = {"instruction": user_text, "output": assistant_text}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============== MEMORY SYSTEM ===============

conversations = load_conversations()  # Load all topics on startup


def all_memory_pairs() -> List[dict]:
    """Flatten all conversations into instruction/output pairs."""
    mem = []
    for conv in conversations:
        for u, b in conv.get("turns", []):
            mem.append({"instruction": u, "output": b})
    return mem


def build_messages_from_memory() -> List[dict]:
    """Build message context including system prompt and recent memory."""
    system_msg = {
        "role": "system",
        "content": (
            "You are JD, an AI Assistant created by Kunal Debnath Sir. "
            "You remember previous chats and stay friendly and concise."
        ),
    }
    
    mem_pairs = all_memory_pairs()
    mem_texts = []
    
    # Include last 50 conversation turns as context
    for p in mem_pairs[-50:]:
        mem_texts.append(f"User: {p['instruction']}\nAssistant: {p['output']}")
    
    block = "\n\n".join(mem_texts) if mem_texts else ""
    
    if block:
        memory_msg = {
            "role": "system",
            "content": "Here is a summary of past conversation turns:\n\n" + block,
        }
        return [system_msg, memory_msg]
    
    return [system_msg]


# =============== UTILITIES ===============

def make_tts(text: str) -> str:
    """Generate text-to-speech audio from text."""
    if not text:
        return ""
    filename = os.path.join(AUDIO_DIR, "jd_reply.mp3")
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"TTS error: {e}")
        return ""


def extract_title_from_history(history: List[List[str]]) -> str:
    """Generate conversation title from first user message."""
    if not history:
        return "New chat"
    first_user = history[0][0].strip()
    if not first_user:
        return "New chat"
    words = first_user.split()
    short = " ".join(words[:10])
    return short + ("…" if len(words) > 10 else "")


# =============== CORE LOGIC ===============

def jd_reply_core(
    user_input: str,
    speak: bool,
    ui_history: List[List[str]],
    conv_id: Optional[str]
) -> Tuple[List[List[str]], Optional[str], Optional[str], List[str]]:
    """
    Process user input and generate AI response.
    
    Args:
        user_input: User's message
        speak: Whether to generate TTS audio
        ui_history: Current conversation history
        conv_id: ID of active conversation or None for new
        
    Returns:
        Tuple of (updated_history, audio_path, conversation_id, topic_titles)
    """
    global conversations

    user_input = user_input.strip()
    if not user_input:
        titles = [c["title"] for c in conversations]
        return ui_history, None, conv_id, titles

    # Build context from all past conversations
    messages = build_messages_from_memory()

    # Add current conversation turns
    for u, b in ui_history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_input})

    # Get AI response
    try:
        res = ollama.chat(model=MODEL_NAME, messages=messages)
        answer = res["message"]["content"]
    except Exception as e:
        answer = f"Ollama error: {e}"

    ui_history = ui_history + [[user_input, answer]]

    # Update or create conversation
    if conv_id is None:
        conv_id = str(uuid.uuid4())
        title = extract_title_from_history(ui_history)
        new_conv = {"id": conv_id, "title": title, "turns": ui_history.copy()}
        conversations.append(new_conv)
    else:
        # Update existing conversation
        found = False
        for conv in conversations:
            if conv["id"] == conv_id:
                conv["turns"] = ui_history.copy()
                conv["title"] = extract_title_from_history(ui_history)
                found = True
                break
        if not found:
            # Fallback: create new if ID missing
            conv_id = str(uuid.uuid4())
            title = extract_title_from_history(ui_history)
            conversations.append({"id": conv_id, "title": title, "turns": ui_history.copy()})

    # Save to disk
    save_conversations(conversations)
    log_flat_pair(user_input, answer)

    # Generate TTS if requested
    audio = make_tts(answer) if speak else None
    titles = [c["title"] for c in conversations]
    
    return ui_history, audio, conv_id, titles


def load_conversation_by_title(title: str) -> Tuple[List[List[str]], Optional[str]]:
    """Load conversation history by title."""
    if not title:
        return [], None
    
    for conv in conversations:
        if conv["title"] == title:
            return conv["turns"], conv["id"]
    
    return [], None


# =============== UI STYLING ===============

CUSTOM_CSS = """
/* =============== GLOBAL + ANIMATED BACKGROUND =============== */
html, body {
  margin: 0;
  height: 100vh;
  overflow: hidden;
  background: #050508;  /* dark grey base */
  color: #e5e7eb;
  font-family: "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  position: relative;
}
html::before {
  content: "";
  position: fixed;
  inset: -30%;
  pointer-events: none;
  z-index: -1;
  background:
    radial-gradient(circle at 15% 20%, rgba(255,255,255,0.08), transparent 55%),
    radial-gradient(circle at 80% 70%, rgba(255,255,255,0.06), transparent 55%),
    radial-gradient(circle at 40% 90%, rgba(255,255,255,0.04), transparent 55%);
  animation: jd-bg-orbit 32s ease-in-out infinite alternate;
  filter: blur(2px);
}
@keyframes jd-bg-orbit {
  0%   { transform: translate3d(0, 0, 0) scale(1); }
  50%  { transform: translate3d(-40px, -25px, 0) scale(1.03); }
  100% { transform: translate3d(35px, 30px, 0) scale(1.02); }
}

/* app container fade-in */
.gradio-container {
  max-width: 100%;
  height: 100vh;
  padding: 0 !important;
  background: transparent;
  animation: jd-app-in 0.45s ease-out;
}
@keyframes jd-app-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

#jd-root {
  display: flex;
  height: 100vh;
}

/* =============== SIDEBAR =============== */
#jd-sidebar {
  width: 210px;
  background: radial-gradient(circle at top left, rgba(15,23,42,0.96), rgba(15,23,42,0.98));
  border-right: 1px solid #111827;
  border-radius: 0 16px 16px 0;
  display: flex;
  flex-direction: column;
  padding: 16px 14px 14px 18px;
  gap: 16px;
  box-shadow:
    12px 0 40px rgba(0,0,0,0.7),
    0 0 0 1px rgba(15,23,42,0.9);
}
#jd-logo {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.32em;
  text-transform: uppercase;
  color: #f9fafb;
}

/* New chat button */
#jd-newchat {
  padding: 8px 10px;
  border-radius: 999px;
  border: 1px solid #4b5563;
  background: radial-gradient(circle at top left, #111827, #020617);
  color: #e5e7eb;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: transform 0.16s ease-out, box-shadow 0.16s ease-out, border-color 0.16s;
}
#jd-newchat:hover {
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 12px 26px rgba(37,99,235,0.6);
  border-color: #60a5fa;
}

/* Conversations list */
#jd-topic-radio > label {
  font-weight: 600;
  letter-spacing: 0.01em;
  margin-bottom: 6px;
}
#jd-topic-radio .wrap.svelte-1ipelgc {
  max-height: calc(100vh - 160px);
  overflow-y: auto;
}
#jd-topic-radio .wrap.svelte-1ipelgc > div {
  background: linear-gradient(135deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
  border-radius: 14px;
  border: 1px solid rgba(55,65,81,0.9);
  padding: 8px 10px;
  margin-bottom: 8px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.7);
  transition:
    transform 0.16s ease-out,
    box-shadow 0.16s ease-out,
    border-color 0.16s ease-out,
    background 0.18s ease-out;
}
#jd-topic-radio .wrap.svelte-1ipelgc > div:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 40px rgba(15,23,42,0.95);
  border-color: rgba(168,85,247,0.9);
  background: radial-gradient(circle at top left, rgba(30,64,175,0.85), rgba(15,23,42,0.98));
}
#jd-topic-radio input[type="radio"] {
  accent-color: #f97316;
}
#jd-topic-radio label {
  font-size: 13px;
  color: #e5e7eb;
}

#jd-sidebar-footer {
  font-size: 11px;
  color: #6b7280;
}

/* =============== MAIN COLUMN =============== */
#jd-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* top bar with animated underline */
#jd-topbar {
  padding: 10px 20px;
  border-bottom: 1px solid #111827;
  font-size: 14px;
  color: #9ca3af;
  position: relative;
}
#jd-topbar span {
  color: #e5e7eb;
  font-weight: 500;
}
#jd-topbar::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: -1px;
  height: 2px;
  background: linear-gradient(90deg,#22c55e,#38bdf8,#a855f7,#22c55e);
  background-size: 260% 100%;
  animation: jd-topbar-glow 10s linear infinite;
  opacity: 0.85;
}
@keyframes jd-topbar-glow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* =============== CHAT AREA =============== */
#jd-chat-wrapper {
  padding: 16px;
  height: 50vh;
}
#jd-chatbot > div {
  box-sizing: border-box;
  padding: 10px 18px 14px 18px;
  height: 100%;
  overflow-y: auto;
  background: radial-gradient(circle at top, rgba(15,23,42,0.95), rgba(15,23,42,0.98));
  border-radius: 18px;
  border: 1px solid #111827;
  box-shadow:
    0 0 0 1px rgba(15,23,42,0.9),
    0 22px 60px rgba(0,0,0,0.9);
}

/* messages */
.message.user,
.message.bot {
  animation: jd-msg-pop 0.2s ease-out;
  font-family: inherit;
}
@keyframes jd-msg-pop {
  from { opacity: 0; transform: translateY(6px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* user bubble */
.message.user {
  background: #e5e7eb !important;
  color: #020617 !important;
  max-width: 80%;
  min-width: 160px;
  padding: 10px 18px !important;
  border-radius: 18px !important;
  display: inline-block;
  line-height: 1.35 !important;
  word-break: break-word;
  margin-left: auto;
  box-shadow: 0 10px 26px rgba(15,23,42,0.75);
}
.message.user * {
  color: inherit !important;
  font-family: inherit !important;
  line-height: 1.35 !important;
}

/* bot bubble */
.message.bot {
  background: radial-gradient(circle at top left, #1f2937, #020617) !important;
  color: #e5e7eb !important;
  max-width: 80%;
  padding: 10px 18px !important;
  border-radius: 18px !important;
  border: 1px solid rgba(55,65,81,0.9);
  backdrop-filter: blur(12px);
  box-shadow: 0 14px 38px rgba(15,23,42,0.95);
  line-height: 1.4;
  transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
}
.message.bot:hover {
  transform: translateY(-1px);
  box-shadow: 0 20px 50px rgba(15,23,42,1);
}

/* =============== INPUT AREA =============== */
#jd-input-bar {
  margin-top: 10px;
  padding: 12px 20px 16px 20px;
  border-top: 1px solid #111827;
  background: linear-gradient(to top, rgba(2,6,23,0.98), rgba(2,6,23,0.9));
}
#jd-input-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
#jd-input-row textarea {
  border-radius: 999px !important;
  background: #020617;
  border: 1px solid #4b5563;
  color: #ffffff;
  font-size: 14px;
  padding-left: 16px !important;
  padding-right: 16px !important;
  font-family: inherit;
  resize: none;
  max-height: 90px;
  min-height: 48px;
  overflow-y: auto;
  transition: border-color 0.15s ease-out, box-shadow 0.15s ease-out, transform 0.15s ease-out;
}
#jd-input-row textarea::placeholder {
  color: #6b7280;
}
#jd-input-row textarea:focus {
  border-color: #a855f7;
  box-shadow:
    0 0 0 1px rgba(168,85,247,0.9),
    0 0 34px rgba(168,85,247,0.45);
  transform: translateY(-1px);
}

/* send button */
#jd-send-btn {
  border-radius: 999px;
  background: linear-gradient(135deg, #a855f7, #f97316, #e9d5ff);
  background-size: 220% 220%;
  border: none;
  color: #111827;
  font-weight: 600;
  padding: 9px 28px;
  box-shadow: 0 12px 32px rgba(168,85,247,0.65);
  transition: transform 0.16s ease-out, box-shadow 0.16s ease-out, filter 0.16s ease-out, background-position 0.4s ease-out;
}
#jd-send-btn:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
  background-position: 100% 0%;
  box-shadow: 0 18px 44px rgba(168,85,247,0.8);
}
#jd-send-btn:active {
  transform: translateY(0);
  box-shadow: 0 6px 18px rgba(15,23,42,0.9);
}

/* tools row */
#jd-tools-row {
  display: flex;
  gap: 8px;
  font-size: 11px;
  color: #9ca3af;
  margin-top: 8px;
}
.jd-small-btn {
  border-radius: 999px;
  border: 1px solid #4b5563;
  background: radial-gradient(circle at top, #020617, #020617);
  color: #e5e7eb;
  font-size: 11px;
  padding: 4px 12px;
  transition: transform 0.14s ease-out, box-shadow 0.14s ease-out, border-color 0.14s;
}
.jd-small-btn:hover {
  background: #111827;
  transform: translateY(-1px);
  box-shadow: 0 10px 26px rgba(15,23,42,0.9);
  border-color: #a855f7;
}
"""


# =============== GRADIO INTERFACE ===============

def build_ui():
    """Build and configure the Gradio interface."""
    
    with gr.Blocks(css=CUSTOM_CSS, title="JD Assistant") as demo:
        active_conv_id = gr.State(None)
        topic_titles = gr.State([c["title"] for c in conversations])

        with gr.Row(elem_id="jd-root"):
            # Sidebar
            with gr.Column(elem_id="jd-sidebar", scale=0):
                gr.HTML("<div id='jd-logo'>JD</div>")
                new_chat_btn = gr.Button(elem_id="jd-newchat", value="＋  New chat", size="sm")
                topic_radio = gr.Radio(
                    choices=[c["title"] for c in conversations],
                    label="Conversations",
                    elem_id="jd-topic-radio",
                    interactive=True,
                )
                gr.HTML("<div id='jd-sidebar-footer'>Select a topic to view that chat only.</div>")

            # Main
            with gr.Column(elem_id="jd-main", scale=1):
                gr.HTML("<div id='jd-topbar'><span>JD</span> · Chat interface</div>")

                with gr.Column(elem_id="jd-chat-wrapper"):
                    chat = gr.Chatbot(
                        label="",
                        elem_id="jd-chatbot",
                        height=None,
                        type="tuples",
                    )

                with gr.Column(elem_id="jd-input-bar"):
                    with gr.Row(elem_id="jd-input-row"):
                        txt = gr.Textbox(
                            placeholder="Ask JD anything...",
                            show_label=False,
                            lines=3,
                            scale=8,
                        )
                        send_btn = gr.Button("Send", elem_id="jd-send-btn", scale=1)

                    with gr.Row(elem_id="jd-tools-row"):
                        edit_last_btn = gr.Button("✎ Edit last prompt", elem_classes=["jd-small-btn"])
                        copy_last_btn = gr.Button("⧉ Copy last answer", elem_classes=["jd-small-btn"])
                        tts_toggle = gr.Checkbox(value=True, label="JD server voice", scale=1)

        audio_out = gr.Audio(label="JD voice", autoplay=True, visible=False)
        clipboard_text_output = gr.Textbox(visible=False, interactive=False)

        # =============== EVENT HANDLERS ===============

        def on_send(message, history, speak, conv_id, titles):
            new_history, audio, new_conv_id, new_titles = jd_reply_core(
                message, speak, history or [], conv_id
            )
            return (
                new_history,
                "",
                gr.update(value=audio, visible=bool(audio)),
                new_conv_id,
                new_titles,
                gr.update(choices=new_titles, value=new_titles[-1] if new_titles else None),
            )

        def on_new_chat(conv_id, titles):
            return [], "", gr.update(value=None, visible=False), None, titles, gr.update(value=None)

        def on_select_topic(topic_title):
            if not topic_title:
                return [], None
            full_turns, cid = load_conversation_by_title(topic_title)
            if full_turns is None:
                return [], None
            # Show last 5 turns
            last_five = full_turns[-5:]
            return last_five, cid

        def edit_last_prompt(history):
            if not history:
                return ""
            last_user, _ = history[-1]
            return last_user

        def copy_last_answer(history):
            if not history:
                return ""
            _, last_bot = history[-1]
            return last_bot

        # Wire up events
        send_btn.click(
            fn=on_send,
            inputs=[txt, chat, tts_toggle, active_conv_id, topic_titles],
            outputs=[chat, txt, audio_out, active_conv_id, topic_titles, topic_radio],
        )
        txt.submit(
            fn=on_send,
            inputs=[txt, chat, tts_toggle, active_conv_id, topic_titles],
            outputs=[chat, txt, audio_out, active_conv_id, topic_titles, topic_radio],
        )

        new_chat_btn.click(
            fn=on_new_chat,
            inputs=[active_conv_id, topic_titles],
            outputs=[chat, txt, audio_out, active_conv_id, topic_titles, topic_radio],
        )

        topic_radio.input(
            fn=on_select_topic,
            inputs=topic_radio,
            outputs=[chat, active_conv_id],
        )

        edit_last_btn.click(
            fn=edit_last_prompt,
            inputs=[chat],
            outputs=[txt],
        )

        copy_last_btn.click(
            fn=copy_last_answer,
            inputs=[chat],
            outputs=[clipboard_text_output],
        )

        gr.on(
            [clipboard_text_output.change],
            None,
            inputs=[clipboard_text_output],
            outputs=[],
            js="""
            (text) => {
              if (navigator.clipboard && text) {
                navigator.clipboard.writeText(text);
              }
            }
            """,
        )

    return demo


# =============== MAIN ENTRY POINT ===============

def main():
    """Main entry point for the application."""
    print(f"JD Assistant starting...")
    print(f"Loaded {len(conversations)} conversation topics from storage")
    print(f"Storage directory: {BASE_DIR}")
    
    # Check if Ollama is running
    try:
        ollama.list()
        print(f"✓ Ollama is running")
        print(f"✓ Using model: {MODEL_NAME}")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        print("Please ensure Ollama is installed and running:")
        print("  - Install: https://ollama.ai/download")
        print(f"  - Pull model: ollama pull {MODEL_NAME}")
        print("  - Start server: ollama serve")
        return
    
    # Build and launch UI
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
