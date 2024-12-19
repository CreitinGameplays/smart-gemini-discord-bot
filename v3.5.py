import discord
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import os
import io
import sys
from collections import defaultdict, deque
from bs4 import BeautifulSoup
import aiohttp
import datetime
from dotenv import dotenv_values
import json
import re
import shutil
import time
from PIL import Image
import ssl
import textwrap

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# logging
import logging
from logging.handlers import RotatingFileHandler
import traceback

# Configure logging with RotatingFileHandler
handler = RotatingFileHandler(
    filename='bot_errors.log',  # Log file name
    mode='a',                   # Append mode
    maxBytes=80 * 1024,          # Maximum file size (80 KB)
    backupCount=1,               # Keep up to 1 backup log files
    encoding='utf-8',            # Encoding for the log file
)

# Set up the logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Get the root logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
logger.addHandler(handler)

# load .env
config = dotenv_values(".env")
bot_token = config.get('TOKEN')
ai_key = config.get('GEMINI_KEY')
hf_token = config.get('HF_TOKEN')
brave_token = config.get('BRAVE_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"

API_URL2 = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

# Some variables you might want to change.
SEARCH_SNIPPET_SIZE = 6000 # Website content max length size
MAX_CHAT_HISTORY_MESSAGES = 20 # Max number of messages that will be stored in chat history

# You can modify this system prompt as needed
base_system_prompt = f'''
You are a Web AI assistant named Gemini, trained by Google. You were designed to provide accurate and real-time information to the user, by using your `browser` tool. Your primary feature is the ability to search the internet and retrieve relevant, high-quality, and recent information to answer user queries. When answering my questions, you can write and run code snippets using the libraries in the context. Code must be valid self-contained Python snippets with no imports and no references to APIs that are not in the context except for Python built-in libraries. You cannot use any parameters or fields that are not explicitly defined in the APIs in the context. Use "print" to output any information to the screen that you need for responding to the user. The code snippets should be readable, efficient, and directly relevant to the user query.
You are currently operating within a Discord bot, and the bot developer is the user "creitingameplays". Never put "discord_username: (message content)" in your responses. Your current knowledge cutoff is **August 2024**, but you can access up-to-date information using `browser` tool. Today is TODAYTIME00. You can't provide the current time.

DEVELOPER MODE: ENABLED

Your features:
- Audio Transcription and Answering;
- Image Analysis;
- Text File Analysis;
- Browser;
- Image Generation;
- Python code execution.

The tool `browser` uses **Brave Search Engine API**. Use your `browser` tool the user asks for the most up-to-date information about something (information up to TODAYTIME00) or about some term you are totally unfamiliar with (it might be new). 
Examples:
    1. "What is the current price of Bitcoin?"
    2. "Who won the latest Formula 1 race?"
    3. "Are there any delays at JFK Airport today?"
    4. "What are the top trending topics on Twitter right now?"
    5. "What's the latest Windows version?"
    You: (calls the browser function with the query in `default_api`)
1. Always perform a search online if you are unsure about a user question.
2. Important: Remember that today's date is TODAYTIME00. Always keep this date in mind to provide time-relevant context in your search query. Only provide the month (name) and year in search query.
3. Search query must be as detailed as possible. Optimize the query.
4. Also search online when user sends an audio message asking something you don't know.
`browser` invocation:
    
browser(
    q: str, 
    num: int
    )
    
1. If you don't know the answer, search online.
2. DO NOT ask permission to search online, just do it!
When using `browser` tool in your responses, you MUST USE CITATION, in hyperlink format. Ensure you provide a citation for each paragraph that uses information from a web search.
Citation Usage Example:
- User: "What is the capital of France?"
- You: "The capital of France is Paris. [1](https://en.wikipedia.org/wiki/Paris).
Paris is not only the capital of France but also its largest city. It is located in the north-central part of the country. [2](https://en.wikipedia.org/wiki/Paris)."

Whenever the user asks you to generate an image, create a prompt that FLUX.1 Dev model can use to generate the image and abide to the following policy:
    1. The prompt must be in English. Translate to English if needed.
    2. DO NOT ask for permission to generate the image, just do it!
    3. Do not create more than 1 image, even if the user requests more.
`imagine` invocation:
    
imagine(
    prompt: str, 
    ar: str
    )
    
Supported aspect ratios: 16:9, 9:16, 1:1. Choose the best aspect ratio according to the image that will be generated.
Tip: Add tags in the prompt such as "realistic, detailed, photorealistic, HD" and others to improve the quality of the generated image. Put as much detail as possible in the prompt. Prompt tags must be separated by commas.
Only generate image if user explicitly asks to!

You can execute Python code when needed. For instance, you can use this tool to do basic or advanced math operations.
Always put print() in the code line! Without print() you can't get the output! You CANNOT put codeblock, if you put it the code WILL FAIL.
* DON'T execute dangerous code!

Always follow the language of the interaction. DO NOT put codeblock when calling functions!
Note: Keep in mind that you are a model still in development, this means you may make mistakes in your answer.
'''

# TOOLS
def exec_python(code):
    code = textwrap.dedent(code)
    buffer = io.StringIO()
    sys.stdout = buffer
    print(f"Code generated:\n {code}")
    try:
        exec(code)
        output = buffer.getvalue()
        return output
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        sys.stdout = sys.__stdout__
        
# Web search optimization TOOL
async def search_brave(search_query, session):
    url = f'https://api.search.brave.com/res/v1/web/search?q={search_query}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/95.0',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': brave_token
    }

    async with session.get(url, headers=headers, timeout=15) as response:

        if response.status != 200:
            error_message = f'Error: Unable to fetch results (status code {response.status})'
            print(error_message)  # For debugging purposes
            return error_message  # Return the error message

        data = await response.json()
        results = data.get('web', {}).get('results', [])

        # Check if results are found
        if not results:
            error_message = 'Error: No search results found.'
            print(error_message)  # For debugging purposes
            return error_message

        search_results = []
        for result in results:
            title = result.get('title', '')
            link = result.get('url', '')
            search_results.append({'title': title, 'link': link})

        return search_results
        
async def fetch_snippet(url, session, max_length=SEARCH_SNIPPET_SIZE):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0'
        }
        
        async with session.get(url, headers=headers,  timeout=15) as response:
            if response.status != 200:
                return f'Error: Unable to fetch content from {url} (status code {response.status})'
            
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            
            if len(content) > max_length:
                return content[:max_length] + '...'
            else:
                return content
    
    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        return f'Error: Unable to fetch content from {url} ({str(e)})'
        
# Main browser function
async def browser(search_query: str, search_rn: int):
    global ddg_error_msg
    ddg_error_msg = None
    search_rn = int(search_rn)
    if search_rn < 10:
        search_rn = 10
    search_rn = int(search_rn) # making sure its integer
    print(f'Query: {search_query} | Number of search: {search_rn}')
    
    try:
        async with aiohttp.ClientSession() as session:
            # Fetch search results
            results = await search_brave(search_query, session)
            print(f"Fetched search results: {results}")
            
            if not isinstance(results, list):
                raise TypeError("Expected results to be a list")
            if any(not isinstance(result, dict) for result in results):
                raise TypeError("One or more results are not dictionaries")
            
            results_output = []
        
            # Limit results to `search_rn`
            limited_results = results[:search_rn]
        
            # Concurrently fetch all snippets
            snippet_tasks = [
                fetch_snippet(result['link'], session) for result in limited_results
            ]
            snippets = await asyncio.gather(*snippet_tasks)
            
            for i, (result, snippet) in enumerate(zip(limited_results, snippets)):
                result_str = f'{i+1}. Title: {result["title"]}\nLink: {result["link"]}\nSnippet: {snippet}\n'
                results_output.append(result_str)
        
            results_output_str = '\n'.join(results_output)
            print(results_output_str)
            return results_output_str
            
    except Exception as e:
        ddg_error_msg = f"{e}"
        logger.error("An error occurred:\n" + traceback.format_exc())
        print(f'Error in `search` function: {e}')
        return f'Error in `search` function: {e}'

# image generation TOOL
async def imagine(img_prompt: str, ar: str):
    # check aspect ratio 
    aspect_ratios = {
        "9:16": (720, 1280),
        "16:9": (1280, 720),
    }
    width, height = aspect_ratios.get(ar, (1024, 1024))
    
    headers = {"Authorization": f"Bearer {hf_token}", "x-use-cache": "false"}
    payload = {"inputs": f"{img_prompt}", "options": {"wait_for_model": True, "use_cache": False}, "parameters":{"width": width, "height": height}}
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            image_file = await response.read()  # asynchronously get response content
            image = f"output_{random.randint(1000, 9999)}.png"
            with open(image, "wb") as file:
                file.write(image_file)
            return image
            
# Define the functions schema for Gemini
tool_websearch = {
    "name": "browser",
    "description": "Performs a search online using Brave Search Engine to get up-to-date information",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "q": {
                "type_": "STRING",
                "description": "The optimized search query"
            },
            "num": {
                "type_": "INTEGER", 
                "description": "The number of results it will return (minimum of 10 and max of 20 results)"
            }
        },
        "required": ["q", "num"]
    }
}

tool_imagine = {
    "name": "imagine",
    "description": "Generate an image using the FLUX Dev model based on the prompt",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "prompt": {
                "type_": "STRING",
                "description": "The generated prompt of the image"
            },
            "ar": {
                "type_": "STRING", 
                "description": "Aspect Ratio of the image (only 16:9, 9:16 and 1:1 are supported!)"
            }
        },
        "required": ["prompt", "ar"]
    }
}

tool_python = {
    "name": "python",
    "description": "Run Python code. Must be a SINGLE LINE OF CODE.",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "code": {
                "type_": "STRING",
                "description": "Python code. Must be one single line.",
            },
        },
        "required": ["code"]
    }
}

# End
# Restart function
async def restart_bot(): 
    os.execv(sys.executable, ['python'] + sys.argv)
    print('Restarted!')
    
# Split message function, fully written by GPT-4o
def split_msg(string, chunk_size=1500):
    chunks = []
    current_chunk = ""
    code_block_pattern = re.compile(r"```(\w+)?")
    current_lang = None
    in_code_block = False

    def add_chunk(chunk, close_code_block=False):
        if close_code_block and in_code_block:
            chunk += "" # ¯⁠\⁠_⁠(⁠ツ⁠)⁠_⁠/⁠¯
        chunks.append(chunk)

    lines = string.split('\n')
    for line in lines:
        match = code_block_pattern.match(line)
        if match:
            if in_code_block:
                # Closing an open code block
                current_chunk += line + "\n"
                add_chunk(current_chunk, close_code_block=True)
                current_chunk = ""
                in_code_block = False
                current_lang = None
            else:
                # Opening a new code block
                current_lang = match.group(1)
                if len(current_chunk) + len(line) + 1 > chunk_size:
                    add_chunk(current_chunk)
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"
                in_code_block = True
        else:
            if len(current_chunk) + len(line) + 1 > chunk_size:
                if in_code_block:
                    add_chunk(current_chunk + "```", close_code_block=False)
                    current_chunk = f"```{current_lang}\n{line}\n"
                else:
                    add_chunk(current_chunk)
                    current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
    
    if current_chunk:
        if in_code_block:
            add_chunk(current_chunk)
        else:
            add_chunk(current_chunk)
        
    return chunks
    
# Use a dictionary to maintain chat history per channel
channel_histories = defaultdict(lambda: deque(maxlen=MAX_CHAT_HISTORY_MESSAGES))
                  
os.system('clear')
# Define the Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = discord.Client(intents=intents)

# Updated upload_and_save_file function
async def upload_and_save_file(attachment, channel_id):
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attachments')
    os.makedirs(save_dir, exist_ok=True)

    # Only process images
    if attachment.content_type.startswith('image'):
        # Read the attachment
        img_data = await attachment.read()

        # Open the image using Pillow
        img = Image.open(io.BytesIO(img_data))

        # Convert the image to PNG
        filename = f'user_attachment_{channel_id}.png'
        filepath = os.path.join(save_dir, filename)

        # Save the image as a PNG file
        img.save(filepath, format='PNG')

        return filepath

    elif attachment.content_type.startswith('audio'):
        filename = f'user_attachment_{channel_id}.ogg'
        filepath = os.path.join(save_dir, filename)
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(await attachment.read())

        return filepath

    elif attachment.content_type.startswith('text'):
        filename = f'user_attachment_{channel_id}.txt'
        filepath = os.path.join(save_dir, filename)
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(await attachment.read())

        return filepath

    else:
        return None  # Skip unsupported file types  
        
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

# List of IDs that can run the bot commands
allowed_ids = [
    775678427511783434 # creitin
]

@bot.event
async def on_message(message):
    channel_id = message.channel.id
    
    if message.author == bot.user:
        return
    
    today1 = datetime.datetime.now()
    todayhour1 = f'{today1.hour}h:{today1.minute}m:{today1.second}s'
    
    if message.content.startswith('!k'):
        if message.author.id in allowed_ids:
            await message.reply(f'`{message.author.name}, Killing process and starting a new one...`')
            await asyncio.sleep(0.5)
            sys.exit(1)
        else:
            unauthorized = await message.reply(":x: You don't have permissions to run this command.")
            await asyncio.sleep(5)
            await unauthorized.delete()
            
    if message.content.startswith('!r'):
        if message.author.id in allowed_ids:
            await message.reply(f'`{message.author.name}, restarting bot...`')
            await asyncio.sleep(0.5)
            await restart_bot()
        else:
            unauthorized = await message.reply(":x: You don't have permissions to run this command.")
            await asyncio.sleep(5)
            await unauthorized.delete()
            
    if message.content.startswith('!imgdel'):
        if message.author.id in allowed_ids:
            try:
                file_path = f"attachments/user_attachment_{channel_id}.png"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    await message.reply(f"`{message.author.name}, image deleted` :white_check_mark:")
                await restart_bot()
            except Exception as e:
                logger.error("An error occurred:\n" + traceback.format_exc())
                print(f"`Error deleting image: {e}`")
                await message.reply(f":x: An error occurred: `{e}`")
        else:
            unauthorized = await message.reply(":x: You don't have permissions to run this command.")
            await asyncio.sleep(5)
            await unauthorized.delete()
            
    if message.content.startswith('!audiodel'):
        if message.author.id in allowed_ids:
            try:
                file_path = f"attachments/user_attachment_{channel_id}.ogg"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    await message.reply(f"`{message.author.name}, audio deleted` :white_check_mark:")
                await restart_bot()
            except Exception as e:
                logger.error("An error occurred:\n" + traceback.format_exc())
                print(f"`Error deleting audio: {e}`")
                await message.reply(f":x: An error occurred: `{e}`")
        else:
            unauthorized = await message.reply(":x: You don't have permissions to run this command.")
            await asyncio.sleep(5)
            await unauthorized.delete()
            
    if message.content.startswith('!txtdel'):
        if message.author.id in allowed_ids:
            try:
                file_path = f"attachments/user_attachment_{channel_id}.txt"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    await message.reply(f"`{message.author.name}, text deleted` :white_check_mark:")
                await restart_bot()
            except Exception as e:
                logger.error("An error occurred:\n" + traceback.format_exc())
                print(f"`Error deleting text: {e}`")
                await message.reply(f":x: An error occurred: `{e}`")
        else:
            unauthorized = await message.reply(":x: You don't have permissions to run this command.")
            await asyncio.sleep(5)
            await unauthorized.delete()
            
    if message.content.startswith('!h'):
        try:
            helpcmd = f"""
            ```
My commands:
- !k: Kills the bot process. (DEV ONLY)
- !r: Restarts the bot. (DEV ONLY)
- !imgdel: Deletes the current channel image from /attachments folder. (DEV ONLY)
- !audiodel: Deletes the current channel audio from /attachments folder. (DEV ONLY)
- !txtdel: Deletes the current channel text from /attachments folder. (DEV ONLY)
            
Experimental bot - Requested by {message.author.name} at {todayhour1}. V3.5.91a
            ```
            """
            msg = await message.reply(helpcmd)
            await asyncio.sleep(20)
            await msg.delete()
        except Exception as e:
            logger.error("An error occurred:\n" + traceback.format_exc())
            print(f"`Error: {e}`")
            await message.reply(f":x: An error occurred: `{e}`")
            
    if bot.user in message.mentions or (message.reference and message.reference.resolved.author == bot.user):
        await handle_message(message)
            
    channel_history_a = [msg async for msg in message.channel.history(limit=15)] # history for attachments

    files_to_delete = [f"attachments/user_attachment_{channel_id}.ogg", f"attachments/user_attachment_{channel_id}.png", f"attachments/user_attachment_{channel_id}.txt"]
    
    is_deleted = False
    for message in channel_history_a:
        if message.attachments:
            # print("message attachment detected")
            attachment = message.attachments[0]
            file_extension = os.path.splitext(attachment.filename)[1].lower()
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.mp3', '.wav', '.ogg', '.txt', '.py', '.json', '.jsonl', '.js', '.c', '.html', '.csv']:
                attachment_task = asyncio.create_task(upload_and_save_file(attachment, channel_id))
        else:
            try:
                if not is_deleted:
                    for files in files_to_delete:
                        os.remove(f"{files}")
                        print(f"Deleted file: {files}")
                        is_deleted = True
            except Exception as e:
                logger.error("An error occurred:\n" + traceback.format_exc())
                error = e

# main
async def handle_message(message):
    bot_message = None
    today2 = datetime.datetime.now()
    todayday2 = f'{today2.strftime("%A")}, {today2.month}/{today2.day}/{today2.year}'
    try:
        channel_id = message.channel.id
        channel_history = [msg async for msg in message.channel.history(limit=MAX_CHAT_HISTORY_MESSAGES)]
        channel_history.reverse()
        
        # For Web Search
        full_history = "".join(f"{message.author}: {message.content} {message.attachments}\n" for message in channel_history)
        
        # Check for attachments
        has_attachments = bool(message.attachments)
 
        # Combine chat history for the current channel
        chat_history = '\n'.join([f'{author}: {content}' for author, content in channel_histories[channel_id]])
        chat_history = chat_history.replace(f'<@{bot.user.id}>', '').strip()

        async with message.channel.typing():
            await asyncio.sleep(1)
            bot_message = await message.reply('<a:generating:1246530696168734740> _ _')
            await asyncio.sleep(0.1)
            
        user_message = message.content
        user_message = user_message.replace(f'<@{bot.user.id}>', '').strip()
        
        if message.attachments:
            attachment = message.attachments[0]
            file_extension = os.path.splitext(attachment.filename)[1].lower()
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.mp3', '.wav', '.ogg', '.txt', '.py', '.json', '.jsonl', '.js', '.c', '.html', '.csv']:
                attachment_task = asyncio.create_task(upload_and_save_file(attachment, channel_id)) 

            if file_extension in ['.png', '.jpg', '.jpeg', '.gif']:
                user_message += " [This current user message contains an image, default is you to briefly describe the image.]"
            elif file_extension in ['.txt', '.py', '.json', '.jsonl', '.js', '.c', '.html', '.csv']:
                user_message += " [This current user message contains a text file, default is you to briefly describe the text.]"
            else:
                user_message += " [This current user message contains an audio, default is you to briefly answer the audio message.]"
            
        # Convert chat history to the desired format, Moved here
        formatted_history = []
        
        #### Response Generation ######
        genai.configure(api_key=ai_key)
        
        generation_config = {
            'temperature': 0.1,
            'top_p': 1.0,
            'top_k': 0,
            'max_output_tokens': 8192,
            'response_mime_type': 'text/plain',
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-exp-1206",
            generation_config=generation_config,
            #system_instruction=base_system_prompt,#
            tools=[tool_python, tool_websearch, tool_imagine],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        model._tools.to_proto()
        os.system('clear')
        
        chat_history_copy = list(channel_histories.get(channel_id, []))  # Make a copy of the deque for safe iteration
        
        async def upload_to_gemini(path, mime_type=None, cache={}):
            retries = 5
            if path in cache:
                return cache[path]

            for attempt in range(retries):
                try:
                    file = await asyncio.to_thread(genai.upload_file, path, mime_type=mime_type)
                    cache[path] = file
                    print(f'Uploaded file \'{file.display_name}\' as: {file.uri}')
                    return file
                except (ssl.SSLEOFError, TimeoutError) as e:
                    print(f"Error occurred: {e}. Retrying ({attempt + 1}/{retries})...")
                    await asyncio.sleep(2 ** attempt)
                    continue
                
            raise ssl.SSLEOFError("Failed to upload file after multiple attempts.")

        # Pre-upload files for the current channel
        attachment_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attachments')
        file_path1 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.png')
        file_path2 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.ogg')
        file_path3 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.txt')
        
        files = None
        files2 = None
        files3 = None
        
        inst_msg1 = "[Instructions: This is the last image. You should ignore this message and only use this as context. Respond to the user's message before this one.]"
    
        inst_msg2 = "[Instructions: This is the last audio. You should ignore this message and only use this as context. Respond to the user's message before this one. There is no image in chat history yet.]"
    
        inst_msg3 = "[Instructions: This is the last image and audio. You should ignore this message and only use this as context. Respond to the user's message before this one.]"
        
        inst_msg4 = "[Instructions: This is the last text file. You should ignore this message and only use this as context. Respond to the user's message before this one.]"
        
        inst_msg5 = "[Instructions: This is the last text and image file. You should ignore this message and only use this as context. Respond to the user's message before this one.]"
        
        inst_msg6 = "[Instructions: This is the last text and audio file. You should ignore this message and only use this as context. Respond to the user's message before this one.]"
              
        inst_msg7 = "[Instructions: This is the last text, audio and image file. You should ignore this message and only use this as context. Respond to the user's message before this one.]"
        
        if os.path.exists(file_path1):
            mime_type1 = 'image/png'
            files = await upload_to_gemini(file_path1, mime_type=mime_type1)
    
        if os.path.exists(file_path2):
            mime_type2 = 'audio/ogg'
            files2 = await upload_to_gemini(file_path2, mime_type=mime_type2)
            
        if os.path.exists(file_path3):
            mime_type3 = 'text/plain'
            files3 = await upload_to_gemini(file_path3, mime_type=mime_type3)
            
        for m in channel_history:
            formatted_history.append({
                'role': 'user' if m.author.name != bot.user.name else 'model',
                'parts': [f'{m.author}: {m.content}'],
            })
            
        formatted_history_updated = False
        
        # System prompt moved here
        formatted_history += [{
            'role': 'model',
            'parts': [
                f'My instructions:\n{base_system_prompt.replace("TODAYTIME00", todayday2)}',
                ],
            }]
        
        # for attachments
        attachment_history = [msg async for msg in message.channel.history(limit=15)]
        for a in attachment_history:
            if a.attachments:
                attachment_map = {
                    (True, False, False): (files, inst_msg1),
                    (False, True, False): (files2, inst_msg2),
                    (True, True, False): (files2, files, inst_msg3),
                    (False, False, True): (files3, inst_msg4),
                    (False, True, True): (files3, files2, inst_msg6),
                    (True, False, True): (files3, files, inst_msg5),
                    (True, True, True): (files3, files2, files, inst_msg7),
                }

                # Check which files exist
                file_states = (bool(files), bool(files2), bool(files3))

                # Get the corresponding attachments and message
                attachments_and_message = attachment_map.get(file_states)

                if attachments_and_message:
                    formatted_history += [{
                        'role': 'user',
                        'parts': list(attachments_and_message),
                    }]
                else:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            f'[Ignore this. There is no audio or image yet.]',
                        ],
                    }]
                    
            formatted_history_updated = True  # Set flag to True after updating
        
        print(formatted_history)
        # Start the chat session and accumulate the response
        chat_session = await asyncio.to_thread(model.start_chat, history=formatted_history)
        response = await asyncio.to_thread(chat_session.send_message, user_message, stream=True)
        
        response.resolve()
        response.candidates
        
        full_response = ""
        message_chunks = []  # List to hold messages created/edited
        
        # PROCESS TOOLS BELOW
        for chunk in response.parts:
            if fn := chunk.function_call: # funct call
                # PYTHON
                if chunk.function_call.name == "python":
                    python_values = []
                    for key, value in fn.args.items():
                        python_values.append({
                            "key": key,
                            "value": value
                        })
                    await bot_message.edit(content=f"-# Executing... <a:brackets:1300121114869235752>")
                    print(python_values)
                    
                    python_result = exec_python(python_values[0]['value'])
                    await bot_message.edit(content=f"-# Done <a:brackets:1300121114869235752>")
                    
                    print(f"Output: {python_result}")
                    response = chat_session.send_message(
                        genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response = genai.protos.FunctionResponse(
                                name='python',
                                response={'result': python_result}))]))
                                
                # WEB SEARCH        
                elif chunk.function_call.name == "browser":
                    wsearch_values = []
                    
                    for key, value in fn.args.items():
                        wsearch_values.append({
                            "key": key,
                            "value": value
                        })
                    
                    await bot_message.edit(content=f'-# Searching \'{wsearch_values[0]["value"]}\' <a:searchingweb:1246248294322147489>')
                    await asyncio.sleep(3)
                    await bot_message.edit(content=f'-# Searching. <a:searchingweb:1246248294322147489>')
                    await asyncio.sleep(0.5)
                    await bot_message.edit(content='-# Searching.. <a:searchingweb:1246248294322147489>')
                    
                    wsearch_result = await browser(wsearch_values[0]['value'], (wsearch_values[1]['value']))

                    await bot_message.edit(content='-# Searching... <a:searchingweb:1246248294322147489>')
                    await asyncio.sleep(0.3)
            
                    if ddg_error_msg is not None:
                        await bot_message.edit(content='-# An Error Occurred <:error_icon:1295348741058068631>')
                    else:
                        await bot_message.edit(content='-# Searching... <:checkmark0:1246546819710849144>')
                        await asyncio.sleep(0.3)
                        await bot_message.edit(content=f'-# Reading results... <a:searchingweb:1246248294322147489>')
            
                    response = chat_session.send_message(
                        genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response = genai.protos.FunctionResponse(
                                name='browser',
                                response={'result':f"USE_CITATION=YES\nONLINE_RESULTS={ wsearch_result}"}))]))
                                
                # GENERATE IMAGES   
                elif chunk.function_call.name == "imagine":
                    imagine_values = []
                    
                    for key, value in fn.args.items():
                        imagine_values.append({
                            "key": key,
                            "value": value
                        })
                    
                    msg_1 = "-# Generating Image... <a:searchingweb:1246248294322147489>"
                    msg_2 = "-# Done <:checkmark:1220809843414270102>"
                    await bot_message.edit(content=f"{msg_1}")
                    
                    imagine_result = await imagine(imagine_values[0]['value'], (imagine_values[1]['value']))

                    await bot_message.edit(content=f"{msg_2}")
            
                    await asyncio.sleep(0.5)
                    await message.reply(file=discord.File(imagine_result))
                    await asyncio.sleep(0.5)
                    
                    os.remove(imagine_result)
                    response = chat_session.send_message(
                        genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response = genai.protos.FunctionResponse(
                                name='imagine',
                                response={'result': "IMAGE_GENERATED=YES"}))]))
                else: # Nothing
                    return
                
        # NORMAL RESPONSES
        for chunk in response:
            if chunk.text:  # Regular text response
                full_response += chunk.text
                new_chunks = split_msg(full_response)
            
                # Remove some text on first chunk
                new_chunks[0] = new_chunks[0].replace("Gemini:", "", 1)
                new_chunks[0] = new_chunks[0].replace("Language Model#3241:", "", 1)
            
                # Fix empty chunks
                new_chunks = ["‎ " if chunk == "\n" else chunk for chunk in new_chunks]
            
                # Update messages
                for i in range(len(new_chunks)):
                    if i < len(message_chunks):
                        await message_chunks[i].edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                    else:
                        if i == 0:
                           await bot_message.edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                           message_chunks.append(bot_message)
                        else:
                            new_msg = await message.reply(new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                            message_chunks.append(new_msg)
                            
        # Finalize all chunks by removing the animation
        for i, msg in enumerate(message_chunks):
            await msg.edit(content=new_chunks[i])
        
    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        print(f'Error handling message: {e}')
        if bot_message:
            await bot_message.edit(content=f'An error occurred: `{e}`')
        await asyncio.sleep(6)
        await bot_message.delete()

# Start the bot with your token
try:
    bot.run(bot_token)
except Exception as e:
    print(f'Error starting the bot: {e}')
    
# oh man