# Smart Gemini Discord Bot

This is a Discord bot built using the [py-cord](https://github.com/Pycord-Development/pycord)

## It can

- Web Search
- Image Analysis
- Audio Transcription
- Generate images
- Read text files
- Execute Python code snippet (`v3.5.py` and above)
  
## Requirements
- Your bot token (Get it on Discord Developers Portal)
- Groq API key (it's free)
- Google AI Studio API key (it's also free)
- HuggingFace token (free)
- A functional human brain (expensive)

`v3.5` and above:
- Brave Search API key (free but requires cc)

Note: Files with `dalle` in the name uses DALL•E 3 for generating images, means you'll need an OpenAI API Key + waste some money with DALL•E 3. The rest uses free image generation model.

`v3.5.py` and above do not require Groq API Key.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/CreitinGameplays123/smart-gemini-discord-bot.git
    cd smart-gemini-discord-bot
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables by creating a `.env` file in the project directory and adding your keys:
    ```sh
    TOKEN=your_discord_bot_token
    GEMINI_KEY=your_gemini_api_key
    GROQ_KEY=your_groq_api_key
    HF_TOKEN=your_hf_token
    BRAVE_TOKEN=your_brave_token
    ```

## Usage

Run the bot:
```sh
chmod +x start.sh
./start.sh
```

## Bot Commands

- `!del` - Deletes the current channel chat history from the JSON file (not available in `v3.2.py`).
- `!k` - Kills the bot process.
- `!r` - Restarts the bot.
- `!imgdel` - Deletes the current channel image from the `attachments` folder.
- `!audiodel` - Deletes the current channel audio from the `attachments` folder.
- `!txtdel` - Deletes the current channel text from the `attachments` folder.
- `!h` - Displays the help command with available bot commands.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



