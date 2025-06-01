# Smart Gemini Discord Bot

This is a Discord bot built using the [py-cord](https://github.com/Pycord-Development/pycord)

## It can

- Web Search
- Image Analysis
- Audio Transcription
- Generate images
- Read text files
- Execute Python code snippet
- And more!
  
## Requirements
- Your bot token (Get it on Discord Developers Portal)
- ~~Groq API key (it's free)~~ v3.5 and up doesn't need this
- Google AI Studio API key (it's also free)
- ~~HuggingFace token (free)~~ v3.5 and up doesn't need this
- A functional human brain (expensive)

`v3.5` and above:
- Brave Search API key (free but requires cc)
- New: Google Cloud account with Imagen 3 access (needs 💵)

`v4.0`:
- Process YouTube videos (native, free)
  
Note: Files with `dalle` in the name uses DALL•E 3 for generating images, means you'll need an OpenAI API Key + waste some money with DALL•E 3. The rest uses free image generation model, except `v3.5.py` and up.

`v3.5.py` and above do not require Groq API Key.

## Setup Google Cloud account (v3.5.py and up)
- In terminal, run the following commands:
```
snap install google-cloud-cli
```

```
gcloud auth application-default login
```
- Then copy the link in terminal -> Choose your Google Cloud account -> Allow everything -> Copy the code it will display -> Go back to the terminal and paste the code -> then run:
```
gcloud config set project YOUR_GOOGLE_CLOUD_PROJECT_ID
```
- Done!

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
    GCP_PROJECT=your_google_cloud_project
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

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.



