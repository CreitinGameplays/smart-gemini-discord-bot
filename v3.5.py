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
from groq import Groq
import shutil
import time
from PIL import Image

import google.generativeai as genai

from google.generativeai.types import HarmCategory, HarmBlockThreshold
from duckduckgo_search import DDGS

# load .env
config = dotenv_values(".env")
bot_token = config.get('TOKEN')
ai_key = config.get('GEMINI_KEY')
groq_token = config.get('GROQ_KEY')
hf_token = config.get('HF_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

API_URL2 = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

# Some variables you might want to change.
SEARCH_SNIPPET_SIZE = 5000 # Website content max length size
MAX_CHAT_HISTORY_MESSAGES = 25 # Max number of messages that will be stored in chat history

# Get today's date and format it
today = datetime.datetime.now()
todayday = f'{today.strftime("%A")}, {today.month}/{today.day}/{today.year}'
todayhour = f'{today.hour}h:{today.minute}m'

# Base system prompt without web search results
# You can modify this system prompt as needed
base_system_prompt = f'''
You are Gemini, a large language model trained by Google AI, based on the Gemini 1.5 Flash model. We are interacting on a Discord chat. This means most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs. Never use emojis, unless explicitly asked to. 
You are operating within a Discord bot, and the bot developer is the user "creitingameplays". Never put "discord_username: (message content)" in your answers.
Name: Gemini
Knowledge cutoff: Unknown
Current date: {todayday}
You cannot provide the current time.
You can see usernames from the chat history.

Image Generation Tool: ENABLED
DEVELOPER MODE: ENABLED
Tools: ENABLED

// YOUR TOOLS:
- Audio Transcription and Answering;
- Image Analysis;
- Text File Analysis;
- Web Search;
- Image Generation.

### Web Browser
// You can access more updated information using your web search feature, which uses **DuckDuckGo Search Engine**.
// VERY IMPORTANT: You MUST ALWAYS use CITATION when using Web Search in your responses, in hyperlink format. Ensure you provide a citation for each paragraph that uses information from a web search.
// If you don't know the answer to something, ask the user if they want you to research online about it.
### Citation Example:
User: "What is the capital of France?"
You: "The capital of France is Paris. [1](https://en.wikipedia.org/wiki/Paris).
Paris is not only the capital of France but also its largest city. It is located in the north-central part of the country. [2](https://en.wikipedia.org/wiki/Paris)."

### Audio capability
If you don't understand the user's audio message, ask them to resend it.
// IMPORTANT: Only analyze audios if the user explicitly asks you to do so. If the user sends an audio message, answer the question in it and don't just transcribe it. Think of it as if the audio is going directly to you and you need to answer its question.
// You should also always prioritize the most recent audio message sent by the user, totally ignoring the chat history.

### Image Analysis capability
Only analyze images if the user explicitly asks you to do so.
// Always prioritize the most recent user-uploaded image.

### Image Generation Capability
// You can generate images directly using "FLUX.1 Schnell" model.
// WHENEVER a description of an image is given, create a prompt that FLUX.1 Schnell can use to generate the image and abide to the following policy:
// 1. The prompt must be in English. Translate to English if needed.
// 2. DO NOT ask for permission to generate the image, just do it!
// 3. Do not create more than 1 image, even if the user requests more.
// 4. Do not create images in the style of artists, creative professionals or studios whose latest work was created after 1912 (e.g. Picasso, Kahlo).
// 5. To generate an image, YOU MUST END your message with the prompt you enhanced, like this: 'generate_img:PROMPT'. Tip: Add tags such as "realistic, detailed, photorealistic, HD" and others to improve the quality of the generated image. Put as much detail as possible in the prompt. When generating an image, ALWAYS MAKE A BRIEF DESCRIPTION of it before generating it.
// 6. NEVER FORGET the 'generate_img' call when generating images.

### Example of your image generation invocation:
\ngenerate_img: A realistic photo of a colorful garden and birds flying, cinematic, 4k, HD

Respond with "I'm sorry but I cannot assist you with that." (or a similar message) if the user's message requests dangerous, harmful, adult content, sensitive topics or sexual activities.

Keep in mind that you are a model still in development, this means you may make mistakes in your answer.
'''

# Web search with audio message
async def user_audio(file_attachment):
    headers = {"Authorization": f"Bearer {hf_token}", "x-use-cache": "false"}
    data = await file_attachment.read()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL2, headers=headers, data=data) as response:
                response_text = await response.json()
                print(response_text)
                return response_text
                
    except Exception as e:
        print(f'Error: {e}')
        error_message = f'Transcription error: {e}'
        return error_message
        
# image generation 
async def generate_img(img_prompt):
    headers = {"Authorization": f"Bearer {hf_token}", "x-use-cache": "false"}
    payload = {"inputs": f"{img_prompt}", "options": {"wait_for_model": True, "use_cache": False}}
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            image_file = await response.read()  # asynchronously get response content
            image = "output.png"
            with open(image, "wb") as file:
                file.write(image_file)
            return image
            
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
        
# check using Llama 3
async def needs_search(message_content, has_attachments, message):
    global search_rn
    client = Groq(api_key=groq_token)
    
    if has_attachments:
        attachment = message.attachments[0]
        if attachment.content_type.startswith('audio'):
            audio_transcription = await user_audio(attachment)
            message_content += f" [User message contains an audio, carefully analyze it - User audio transcription: {audio_transcription}]"
        else:
            message_content += " [User message contains an image - DO NOT web search]"
        
    completion = client.chat.completions.create(
        model='llama-3.2-90b-text-preview',
        messages=[
            {
            "role": "system",
            "content": f"""
You are a helpful AI assistant called Gemini Web Helper. Your knowledge cutoff date is October 2023. Today's date is {todayday}.
Your job is to decide when Gemini needs to do a web search based on chat history below. Chat history is a Discord chat between user and Gemini (Gemini is the language model).
Please carefully analyze the conversation to determine if a web search is needed in order for you to provide an appropriate response to the lastest user message.
Highly recommended searching in the following circumstances:
- User is asking Gemini about current events or something that requires real-time information (weather, sports scores, etc.).
- User is asking Gemini the latest information of something, means they want information until  {todayday}.
- User is asking Gemini about some term you are totally unfamiliar with (it might be new).
- User explicitly asks Gemini to browse or provide links to references.
Just respond with 'YES' or 'NO' if you think the following user chat history requires an internet search, don't say anything else than that.
If you believe a search will be necessary, skip a line and generate a search query that you would enter into the DuckDuckGo search engine to find the most relevant information to help you respond.
Use conversation history to get context for web searches. Your priority is the last user message.
Remember that every web search you perform is stateless, meaning you will need to search again if necessary.
The search query must also be in accordance with the language of the conversation (e.g Portuguese, English, Spanish etc.)
Keep it simple and short. Always output your search like this: SEARCH:example-search-query. Always put the `SEARCH`. Do not put any slashes in the search query. To choose a specific number of search results this will return, skip another line and put it like this: RESULTS:number, example: RESULTS:5. Always put the `RESULTS`, only works like that. Minimum of 5 and maximum of 25 search results, the minimum recommended is 15 search results. THIS IS REQUIRED. First is SEARCH, second is RESULTS.
You should NEVER do a web search if the user's message asks for dangerous, insecure, harmful, +18 (adult content), sexual content and malicious code. Just ignore these types of requests.
Respond with plain text only. Do not use any markdown formatting. Do not include any text before or after the search query. For normal searches, don't include the "site:".
Remember that today's date is {todayday}! Always keep this date in mind to provide time-relevant context in your search query.
Focus on generating the single most relevant search query you can think of to address the user's message. Do not provide multiple queries.
Default is not web searching when user asks the model to generate images.
"""
            },
            {
                'role': 'user',
                'content': f'''
<conversation>
{message_content}
(last message)
</conversation>
'''
            }
        ],
        temperature=0.3,
        max_tokens=1024,
        top_p=1.0,
        stream=False,
        stop=None,
    )
    
    output = completion.choices[0].message.content.strip()
    
    # check if its ok
    os.system("clear")
    print(output)
    # print(message_content)
    # If the output suggests a search is needed, extract the search query
    if output.startswith('YES'):
        search_index = output.find('SEARCH:')
        if search_index != -1:
            search_query = output[search_index + len('SEARCH:'):].strip()
            print(f'Extracted search query: {search_query}')
            
        search_num = output.find('RESULTS:')
        if search_num != -1:
            search_rn = output[search_num + len('RESULTS:'):].strip()
            int(search_rn)
            print(f"Extracted number of results: {search_rn}")
            
            return search_query, search_rn
    
    return None

# Web search optimization
async def search_duckduckgo(search_query, session):
    search_query = search_query[0]
    search_query = search_query.replace("\n", "").strip()
    url = f'https://html.duckduckgo.com/html/search?q={search_query}'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/95.0'
    }
    
    async with session.get(url, headers=headers) as response:
        if response.status != 200:
            return f'Error: Unable to fetch results (status code {response.status})'
        
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        results = soup.find_all('a', class_='result__a')
        search_results = []
        for result in results:
            title = result.get_text()
            link = result['href']
            search_results.append({'title': title, 'link': link})

        return search_results

async def fetch_snippet(url, session, max_length=SEARCH_SNIPPET_SIZE):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0'
        }
        
        async with session.get(url, headers=headers) as response:
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
        return f'Error: Unable to fetch content from {url} ({str(e)})'

async def search(search_query):
    global search_rn
    search_rn = int(search_rn)
    
    # Reuse the same session for both search and snippet fetching
    async with aiohttp.ClientSession() as session:
        # Fetch search results
        results = await search_duckduckgo(search_query, session)
        results_output = []
        
        # Limit results to `search_rn`
        limited_results = results[:search_rn]
        
        # Concurrently fetch all snippets
        snippet_tasks = [
            fetch_snippet(result['link'], session) for result in limited_results
        ]
        snippets = await asyncio.gather(*snippet_tasks)
        
        # Combine search results with fetched snippets
        for i, (result, snippet) in enumerate(zip(limited_results, snippets)):
            result_str = f'{i+1}. Title: {result["title"]}\nLink: {result["link"]}\nSnippet: {snippet}\n'
            results_output.append(result_str)
        
        results_output_str = '\n'.join(results_output)
        print(results_output_str)
        return results_output_str
        
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

# List of IDs that can run the bot commands
allowed_ids = [
    775678427511783434, # creitin
    1205039741754671147 # meow
]

@bot.event
async def on_message(message):
    auto_respond_channel_id = 1252258977555943576
    channel_id = message.channel.id
    
    if message.author == bot.user:
        return

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
            
Experimental bot - Requested by {message.author.name} at {todayhour}. V3.5.05
            ```
            """
            msg = await message.reply(helpcmd)
            await asyncio.sleep(20)
            await msg.delete()
        except Exception as e:
            print(f"`Error: {e}`")
            await message.reply(f":x: An error occurred: `{e}`")
            
    if bot.user in message.mentions or (message.reference and message.reference.resolved.author == bot.user) or message.channel.id == auto_respond_channel_id:
        await handle_message(message)
            
    channel_history_a = [msg async for msg in message.channel.history(limit=15)]

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
                error = e
                #print(error)
        
# main
async def handle_message(message):
    bot_message = None
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

        system_prompt = base_system_prompt

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
        web_search = []
        # Check if a web search is needed
        search_query = await needs_search(full_history, has_attachments, message)
        if search_query:
            # Process search query and edit messages
            var1 = f"{search_query}"
            var2 = var1.split("'")[1].split("RESULTS")[0]
            var2 = var2.replace("\n", "").strip()
            num_results = var1.split("'")[1].split("RESULTS:")[1]
            
            await bot_message.edit(content=f'-# Searching "{var2}" <a:searchingweb:1246248294322147489>')
            await asyncio.sleep(3)
            await bot_message.edit(content=f'-# Searching. <a:searchingweb:1246248294322147489>')
            await asyncio.sleep(0.3)
            await bot_message.edit(content='-# Searching.. <a:searchingweb:1246248294322147489>')
            results = await search(search_query)
            
            await bot_message.edit(content='-# Searching... <a:searchingweb:1246248294322147489>')
            await asyncio.sleep(0.3)
            await bot_message.edit(content='-# Searching... <:checkmark0:1246546819710849144>')
            await asyncio.sleep(0.3)
            await bot_message.edit(content=f'-# Reading {num_results} results... <a:searchingweb:1246248294322147489>')
            
            web_search += [{
                'role': 'model',
                'parts': [
                    f'This below is the web search results:\n{results}',
                ],
            }]
            
        #### Response Generation ######
        genai.configure(api_key=ai_key)
        
        generation_config = {
            'temperature': 0.1,
            'top_p': 1.0,
            'top_k': 1,
            'max_output_tokens': 8192,
            'response_mime_type': 'text/plain',
        }
        
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-002',
            generation_config=generation_config,
            #tools='code_execution',
            system_instruction=system_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        chat_history_copy = list(channel_histories.get(channel_id, []))  # Make a copy of the deque for safe iteration

        async def upload_to_gemini(path, mime_type=None, cache={}):
            if path in cache:
                return cache[path]

            file = await asyncio.to_thread(genai.upload_file, path, mime_type=mime_type)  # Run the blocking upload in a separate thread
            cache[path] = file

            print(f'Uploaded file \'{file.display_name}\' as: {file.uri}')
            return file

        # Pre-upload files for the current channel
        attachment_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attachments')
        file_path1 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.png')
        file_path2 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.ogg')
        file_path3 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.txt')
        
        files = None
        files2 = None
        files3 = None
        
        inst_msg1 = "[Instructions: This is the last image. You should ignore this message and only use this as context. Respond to the user's message before this one. There is no audio in chat history yet.]"
    
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
                
        formatted_history += web_search
        
        formatted_history_updated = False
        # for attachments
        attachment_history = [msg async for msg in message.channel.history(limit=15)]
        for a in attachment_history:
            if a.attachments and not formatted_history_updated:
                if files and not files2 and not files3:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files,
                            f'{inst_msg1}',
                        ],
                    }]
                elif files2 and not files and not files3:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files2,
                            f'{inst_msg2}',
                        ],
                    }]
                elif files and files2 and not files3:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files2,
                            files,
                            f'{inst_msg3}',
                        ],
                    }]
                elif files3 and not files2 and not files:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files3,
                            f'{inst_msg4}',
                        ],
                    }]
                    
                elif files3 and not files and files2:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files3,
                            files2,
                            f'{inst_msg6}',
                        ],
                    }]
                    
                elif files3 and files and not files2:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files3,
                            files,
                            f'{inst_msg5}',
                        ],
                    }]
                    
                elif files3 and files and files2:
                    formatted_history += [{
                        'role': 'user',
                        'parts': [
                            files3,
                            files2,
                            files,
                            f'{inst_msg7}',
                        ],
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
        
        full_response = ""
        message_chunks = []  # List to hold messages created/edited
        generate_img_detected = False
        img_prompt = ""

        # Process the response in real-time
        for chunk in response:
            full_response += chunk.text
            new_chunks = split_msg(full_response)

            # Remove some text on first chunk
            new_chunks[0] = new_chunks[0].replace("Gemini:", "", 1)
            new_chunks[0] = new_chunks[0].replace("Language Model#3241:", "", 1)
            
            # Fix empty chunks
            new_chunks = ["‎ " if chunk == "\n" else chunk for chunk in new_chunks]

            # Check if the first chunk contains "generate_img:"
            if new_chunks and "generate_img:" in new_chunks[0]:
                generate_img_detected = True
                img_prompt_start = new_chunks[0].index("generate_img:") + len("generate_img:")
                img_prompt = new_chunks[0][img_prompt_start:].strip()
                print(f"Extracted prompt: {img_prompt}")
                new_chunks[0] = new_chunks[0][:img_prompt_start - len("generate_img:")].strip()

            # Create or edit messages based on the new chunks
            for i in range(len(new_chunks)):
                if i < len(message_chunks):
                    # Edit existing message if it already exists
                    await message_chunks[i].edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                else:
                    if i == 0:
                        # Edit the initial bot message
                        await bot_message.edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                        message_chunks.append(bot_message)  # Add bot_message to the list
                    else:
                        # Create a new message for additional chunks
                        new_msg = await message.reply(new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                        message_chunks.append(new_msg)

        # If "generate_img:" was detected, call the generate_img function and send the image
        if generate_img_detected:
            generating_msg1 = await message.channel.send(content="-# Generating Image... <a:searchingweb:1246248294322147489>")
            generating_msg2 = "-# Done <:checkmark:1220809843414270102>"
            
            generated_image_path = await generate_img(img_prompt)
            await generating_msg1.edit(content=generating_msg2)
            await asyncio.sleep(0.5)
            
            await message.reply(file=discord.File(generated_image_path))
            await asyncio.sleep(0.5)
            
            os.remove(generated_image_path)
            await generating_msg1.delete()
            
        # Finalize all chunks by removing the animation
        for i, msg in enumerate(message_chunks):
            await msg.edit(content=new_chunks[i])
                # Append the bot's message to the chat history
                    
    except Exception as e:
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