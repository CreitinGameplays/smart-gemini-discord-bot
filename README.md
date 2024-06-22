# Smart Gemini Discord Bot

This is a Discord bot built using the [py-cord](https://github.com/Pycord-Development/pycord)

## It also can

- Web Search (Using Llama 3 70b from Groq)
- Image Analysis
- Audio Transcription

## Requirements
- Your bot token
- Groq API key (it's free)
- Google AI Studio API key (it's also free)
- A functional human brain
  
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/CreitinGameplays123/smart-gemini-discord-bot.git
    cd gemini-discord-bot
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
    ```

## Usage

Run the bot:
```sh
chmod +x start.sh
./start.sh
```

## Bot Commands

- `!del` - Deletes the current channel chat history from the JSON file.
- `!k` - Kills the bot process.
- `!r` - Restarts the bot.
- `!imgdel` - Deletes the current channel image from the `attachments` folder.
- `!audiodel` - Deletes the current channel audio from the `attachments` folder.
- `!h` - Displays the help command with available bot commands.
- 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



