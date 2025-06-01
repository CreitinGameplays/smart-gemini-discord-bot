# Smart Gemini Discord Bot

A powerful Discord bot using [py-cord](https://github.com/Pycord-Development/pycord) and Google Gemini, with advanced web search, image, audio, and code execution features.
[Add the bot!](https://discord.com/oauth2/authorize?client_id=1219407466526146661&scope=bot&permissions=277025704960)
---

## Features

- **Web Search**
- **Image Analysis**
- **Audio Transcription**
- **Image Generation**
- **Text File Reading**
- **Python Code Execution**
- **YouTube Video Processing** (newer `main.py`)
- **Per-user Settings** (model, temperature, mention, etc. in current `main.py`)
- **And more!**

---

## Requirements

- Discord Bot Token ([Discord Developer Portal](https://discord.com/developers/applications))
- Google AI Studio API Key
- For `v3.5+`:
  - Brave Search API Key (free, requires credit card)
  - Google Cloud account with Imagen 3 access (paid)
- For legacy/experimental versions:
  - (Optional) Groq API Key
  - (Optional) HuggingFace Token

> **Note:**  
> - Files with `dalle` in the name use DALL·E 3 (requires OpenAI API Key, paid).
> - `v3.5.py` and above do **not** require Groq or HuggingFace tokens.

---

## Setup: Google Cloud (for Imagen 3, v3.5+)

```sh
snap install google-cloud-cli
gcloud auth application-default login
gcloud config set project YOUR_GOOGLE_CLOUD_PROJECT_ID
```

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/CreitinGameplays123/smart-gemini-discord-bot.git
    cd smart-gemini-discord-bot
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Configure environment variables:**  
   Create a `.env` file in the project root:
    ```sh
    TOKEN=your_discord_bot_token
    GEMINI_KEY=your_gemini_api_key
    BRAVE_TOKEN=your_brave_token
    GCP_PROJECT=your_google_cloud_project
    # (Optional for legacy) GROQ_KEY=your_groq_api_key
    # (Optional for legacy) HF_TOKEN=your_hf_token
    ```

---

## Usage

Run the bot:
```sh
chmod +x bot/start.sh
./bot/start.sh
```

---

## License

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.