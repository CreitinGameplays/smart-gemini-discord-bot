import discord
import aiofiles
import asyncio
import random
import os
import io
import sys
from collections import defaultdict, deque
from bs4 import BeautifulSoup
import aiohttp
import datetime
import json
import re
import shutil
import time
from PIL import Image
import ssl
import textwrap
import base64
from google.oauth2 import service_account
import logging
from logging.handlers import RotatingFileHandler
import traceback

# Gemini SDK (new)
from google.genai import Client, types
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Imagen 3
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# ENV VARS
THE_CREDENTIALS = os.environ.get("GCP_CREDS")
if not THE_CREDENTIALS:
    raise ValueError("Environment variable GCP_CREDS is not set")
decoded_credentials = base64.b64decode(THE_CREDENTIALS).decode("utf-8")
credentials_info = json.loads(decoded_credentials)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Logging
handler = RotatingFileHandler(
    filename='bot_errors.log', mode='a', maxBytes=80 * 1024, backupCount=1, encoding='utf-8',
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
logger.addHandler(handler)

# ENV
bot_token = os.getenv('TOKEN')
ai_key = os.getenv('GEMINI_KEY')
hf_token = os.getenv('HF_TOKEN')
brave_token = os.getenv('BRAVE_TOKEN')
gcp_project = os.getenv('GCP_PROJECT')

SEARCH_SNIPPET_SIZE = 6000
MAX_CHAT_HISTORY_MESSAGES = 20
model_id = "gemini-2.0-flash"
image_model_id = "imagen-3.0-fast-generate-001"

# SYSTEM PROMPT
base_system_prompt = f"""
You are a Web AI assistant named Gemini, trained by Google. You were designed to provide accurate and real-time information to the user, by using your `browser` tool. Your primary feature is the ability to search the internet and retrieve relevant, high-quality, and recent information to answer user queries.
You are currently operating within a Discord bot, and the bot developer is the user "creitingameplays". Never put "discord_username: (message content)" in your responses. Your current knowledge cutoff is August 2024, but you can access up-to-date information using `browser` tool. Today is TODAYTIME00. You can't provide the current time. Your current model ID: `{model_id}`. Your image model ID: `{image_model_id}`.

DEVELOPER MODE: ENABLED

Your features:
- Audio Transcription and Answering;
- Image Analysis;
- Text File Analysis;
- Browser;
- Image Generation;
- Python code execution.

# BROWSER INSTRUCTIONS
The tool `browser` uses **Brave Search Engine API**. Use your `browser` tool when the user asks for the most up-to-date information about something (information up to TODAYTIME00) or about some term you are totally unfamiliar with (it might be new).
Examples:
    1. "What is the current price of Bitcoin?"
    2. "Who won the latest Formula 1 race?"
    3. "Are there any delays at JFK Airport today?"
    4. "What are the top trending topics on Twitter right now?"
    5. "What's the latest Windows version?"
    You: (calls the browser function with the query in `default_api`)
1. Always perform a search online if you are unsure about a user question.
2. Remember that today's date is TODAYTIME00. Always keep this date in mind to provide time-relevant context in your search query. Only provide the month (name) and year in search query.
3. Search query must be as detailed as possible. Optimize the query.
4. Also search online when user sends an audio message asking something you don't know.
1. If you don't know the answer, search online.
2. DO NOT ask permission to search online, just do it!
When using `browser` tool in your responses, you MUST USE CITATION, in hyperlink format. Ensure you provide a citation for each paragraph that uses information from a web search.
Citation Usage Example:
- User: "What is the capital of France?"
- You: "The capital of France is Paris. [1](https://en.wikipedia.org/wiki/Paris).
Paris is not only the capital of France but also its largest city. It is located in the north-central part of the country. [2](https://en.wikipedia.org/wiki/Paris)."
To search specific websites or domains, use "site:<website-domain>" in your query and nothing else!

# IMAGE GENERATION INSTRUCTIONS
Whenever the user asks you to generate an image, create a prompt that `{image_model_id}` model can use to generate the image and abide to the following policy:
    1. The prompt must be in English. Translate to English if needed.
    2. DO NOT ask for permission to generate the image, just do it!
    3. Do not create more than 1 image, even if the user requests more.
Supported aspect ratios: 16:9, 9:16, 1:1. Choose the best aspect ratio according to the image that will be generated.
Tip: Add tags in the prompt such as "realistic, detailed, photorealistic, HD" and others to improve the quality of the generated image. Put as much detail as possible in the prompt. Prompt tags must be separated by commas.
Only generate image if user explicitly asks to!

# CODE EXECTUTION INSTRUCTIONS
You can execute Python code when needed. For instance, you can use this tool to do basic or advanced math operations.
Example:
    1. "Count r's in strawberry word using code?"
    2. "What is 38 * 4 - 5?"
Always put print() in the code line! Without print() you can't get the output! You CANNOT put codeblock in this, if you put it the code WILL FAIL.
* DON'T execute dangerous code!

# ADDITIONAL INSTRUCTIONS
Always follow the language of the interaction. DO NOT put codeblock when calling functions!
Please always skip a line when you are about to write a code in a codeblock.
Keep in mind that you are a model still in development, this means you may make mistakes in your answer.

DO NOT OUTPUT TEXT-ONLY WHEN CALLING FUNCTIONS.
"""

# TOOLS (new SDK format)
tool_websearch = types.Tool(function_declarations=[
    {
        "name": "browser",
        "description": "Performs a search online using Brave Search Engine to get up-to-date information",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "The optimized search query"},
                "num": {"type": "integer", "description": "The number of results (min 15, max 30)"}
            },
            "required": ["q", "num"]
        }
    }
])
tool_imagine = types.Tool(function_declarations=[
    {
        "name": "imagine",
        "description": f"Generate an image using the {image_model_id} model based on the prompt",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The generated prompt of the image"},
                "ar": {"type": "string", "description": "Aspect Ratio (16:9, 9:16, 1:1)"}
            },
            "required": ["prompt", "ar"]
        }
    }
])
tool_python = types.Tool(function_declarations=[
    {
        "name": "python",
        "description": "Run Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code_text": {"type": "string", "description": "Python code."}
            },
            "required": ["code_text"]
        }
    }
])

# Split message function (unchanged)
def split_msg(string, chunk_size=1500):
    chunks = []
    current_chunk = ""
    code_block_pattern = re.compile(r"```(\w+)?")
    current_lang = None
    in_code_block = False

    def add_chunk(chunk, close_code_block=False):
        if close_code_block and in_code_block:
            chunk += ""
        chunks.append(chunk)

    lines = string.split('\n')
    for line in lines:
        match = code_block_pattern.match(line)
        if match:
            if in_code_block:
                current_chunk += line + "\n"
                add_chunk(current_chunk, close_code_block=True)
                current_chunk = ""
                in_code_block = False
                current_lang = None
            else:
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
        add_chunk(current_chunk)
    return chunks

# Chat history
channel_histories = defaultdict(lambda: deque(maxlen=MAX_CHAT_HISTORY_MESSAGES))

# Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = discord.Client(intents=intents)

# File upload helper
async def upload_and_save_file(attachment, channel_id):
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attachments')
    os.makedirs(save_dir, exist_ok=True)
    if attachment.content_type.startswith('image'):
        img_data = await attachment.read()
        img = Image.open(io.BytesIO(img_data))
        filename = f'user_attachment_{channel_id}.png'
        filepath = os.path.join(save_dir, filename)
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
        return None

# Brave search
async def search_brave(search_query, session):
    url = f'https://api.search.brave.com/res/v1/web/search?q={search_query}'
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json',
        'X-Subscription-Token': brave_token
    }
    async with session.get(url, headers=headers, timeout=15) as response:
        if response.status != 200:
            return f'Error: Unable to fetch results (status code {response.status})'
        data = await response.json()
        results = data.get('web', {}).get('results', [])
        if not results:
            return 'Error: No search results found.'
        search_results = []
        for result in results:
            title = result.get('title', '')
            link = result.get('url', '')
            search_results.append({'title': title, 'link': link})
        return search_results

async def fetch_snippet(url, session, max_length=SEARCH_SNIPPET_SIZE):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with session.get(url, headers=headers, timeout=15) as response:
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

async def browser(search_query: str, search_rn: int):
    search_rn = max(10, int(search_rn))
    try:
        async with aiohttp.ClientSession() as session:
            results = await search_brave(search_query, session)
            if not isinstance(results, list):
                return results
            limited_results = results[:search_rn]
            snippet_tasks = [fetch_snippet(result['link'], session) for result in limited_results]
            snippets = await asyncio.gather(*snippet_tasks)
            results_output = []
            for i, (result, snippet) in enumerate(zip(limited_results, snippets)):
                result_str = f'{i+1}. Title: {result["title"]}\nLink: {result["link"]}\nSnippet: {snippet}\n'
                results_output.append(result_str)
            return '\n'.join(results_output)
    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        return f'Error in `search` function: {e}'

# Imagen 3
async def imagine(img_prompt: str, ar: str):
    vertexai.init(project=gcp_project, location="us-central1", credentials=credentials)
    img_info_var = {"is_error": 0, "img_error_msg": "null"}
    generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
    try:
        image_response = generation_model.generate_images(
            prompt=img_prompt,
            number_of_images=1,
            aspect_ratio=ar,
            safety_filter_level="block_some",
        )
        generated_image = image_response[0]
        image_bytes = generated_image._image_bytes
        image = Image.open(io.BytesIO(image_bytes))
        image_filename = f"output_{random.randint(1000, 9999)}.png"
        image.save(image_filename)
        img_info_var.update({"filename": image_filename})
        return img_info_var
    except Exception as e:
        img_info_var.update({"is_error": 1, "img_error_msg": f"{e}"})
        return img_info_var

def exec_python(code):
    code = textwrap.dedent(code)
    print(code)
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        exec(code)
        output = buffer.getvalue()
        print('output:' + output)
        return output
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        sys.stdout = sys.__stdout__

# Restart
async def restart_bot():
    os.execv(sys.executable, ['python'] + sys.argv)

# Discord events and message handler
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')

allowed_ids = [775678427511783434]

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
Experimental bot - Requested by {message.author.name} at {todayhour1}. V4.0.0
            ```
            """
            msg = await message.reply(helpcmd)
            await asyncio.sleep(20)
            await msg.delete()
        except Exception as e:
            logger.error("An error occurred:\n" + traceback.format_exc())
            await message.reply(f":x: An error occurred: `{e}`")
    if bot.user in message.mentions or (message.reference and message.reference.resolved.author == bot.user):
        await handle_message(message)

async def handle_message(message):
    bot_message = None
    today2 = datetime.datetime.now()
    todayday2 = f'{today2.strftime("%A")}, {today2.month}/{today2.day}/{today2.year}'
    try:
        channel_id = message.channel.id
        channel_history = [msg async for msg in message.channel.history(limit=MAX_CHAT_HISTORY_MESSAGES)]
        channel_history.reverse()
        chat_history = '\n'.join([f'{author}: {content}' for author, content in channel_histories[channel_id]])
        chat_history = chat_history.replace(f'<@{bot.user.id}>', '').strip()
        async with message.channel.typing():
            await asyncio.sleep(1)
            bot_message = await message.reply('<a:gemini_sparkles:1321895555676504077> _ _')
            await asyncio.sleep(0.1)
        user_message = message.content.replace(f'<@{bot.user.id}>', '').strip()
        # Attachments
        attachment_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attachments')
        file_path1 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.png')
        file_path2 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.ogg')
        file_path3 = os.path.join(attachment_folder, f'user_attachment_{channel_id}.txt')
        files, files2, files3 = None, None, None
        client = Client(api_key=ai_key)
        config = types.GenerateContentConfig(
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
            tools=[tool_python, tool_websearch, tool_imagine],
            system_instruction=[
                types.Part.from_text(text=base_system_prompt.replace("TODAYTIME00", todayday2))
            ]
        )
        chat_contents = []
        # Add chat history
        for m in channel_history:
            chat_contents.append(
                types.Content(
                    role="user" if m.author.name != bot.user.name else "model",
                    parts=[types.Part.from_text(text=f"{m.author}: {m.content}")]
                )
            )
        # Attachments
        async def upload_to_gemini(path, mime_type=None, cache={}):
            retries = 5
            if path in cache:
                return cache[path]
            for attempt in range(retries):
                try:
                    uploaded_file = client.files.upload(file=path)
                    cache[path] = uploaded_file
                    return uploaded_file
                except Exception as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise Exception(f"Failed to upload file after {retries} attempts: {str(e)}")
        if os.path.exists(file_path1):
            files = await upload_to_gemini(file_path1, mime_type='image/png')
        if os.path.exists(file_path2):
            files2 = await upload_to_gemini(file_path2, mime_type='audio/ogg')
        if os.path.exists(file_path3):
            files3 = await upload_to_gemini(file_path3, mime_type='text/plain')
        # Add files to chat_contents if present
        if files:
            chat_contents.append(types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files.uri, mime_type=files.mime_type),
                    types.Part.from_text(text="[Instructions: This is the last image. Use as context.]")
                ]
            ))
        if files2:
            chat_contents.append(types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files2.uri, mime_type=files2.mime_type),
                    types.Part.from_text(text="[Instructions: This is the last audio. Use as context.]")
                ]
            ))
        if files3:
            chat_contents.append(types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files3.uri, mime_type=files3.mime_type),
                    types.Part.from_text(text="[Instructions: This is the last text file. Use as context.]")
                ]
            ))
        # Add user message
        chat_contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        ))
        # Generate content with Gemini
        response = client.models.generate_content_stream(
            model=model_id,
            contents=chat_contents,
            config=config
        )
        full_response = ""
        message_chunks = []
        async def process_response_text(response, message, bot_message, message_chunks):
            try:
                if response and hasattr(response, 'text') and response.text:
                    full_response = response.text
                    new_chunks = split_msg(full_response) if full_response else []
                    if not new_chunks or not isinstance(new_chunks, list):
                        new_chunks = ["..."]
                    if new_chunks:
                        new_chunks[0] = new_chunks[0].replace("Gemini:", "", 1)
                        new_chunks[0] = new_chunks[0].replace("Language Model#3241:", "", 1)
                        
                    new_chunks = ["‎ " if chunk == "\n" else chunk for chunk in new_chunks]
                    for i in range(len(new_chunks)):
                        try:
                            if i < len(message_chunks):
                                if message_chunks[i]:
                                    await message_chunks[i].edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                            else:
                                if i == 0 and bot_message:
                                    await bot_message.edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                                    message_chunks.append(bot_message)
                                else:
                                    new_msg = await message.reply(new_chunks[i] + " <a:generatingslow:1246630905632653373>")
                                    message_chunks.append(new_msg)
                        except Exception as e:
                            print(f"Error updating message {i}: {e}")
                            continue
                    for i in range(len(message_chunks)):
                        try:
                            if i < len(new_chunks) and message_chunks[i]:
                                await message_chunks[i].edit(content=new_chunks[i])
                        except Exception as e:
                            print(f"Error finalizing message {i}: {e}")
                    return new_chunks
            except Exception as e:
                print(f"Error in process_response_text: {e}")
                return None
        # Process Gemini response
        for chunk in response:
            try:
                if chunk.function_calls:
                    fn = chunk.function_calls[0]
                    # PYTHON
                    if fn.name == "python":
                        code_text = fn.args.get('code_text', '')
                        await bot_message.edit(content=f"-# Executing... <a:brackets:1300121114869235752>")
                        python_result = exec_python(code_text)
                        await bot_message.edit(content=f"-# Done <a:brackets:1300121114869235752>")
                        function_response_part = types.Part.from_function_response(
                            name="python",
                            response={"result": python_result}
                        )
                        chat_contents.append(types.Content(
                            role="model",
                            parts=[types.Part(function_call=fn)]
                        ))
                        chat_contents.append(types.Content(
                            role="user",
                            parts=[function_response_part]
                        ))
                        response = client.models.generate_content(
                            model=model_id,
                            contents=chat_contents,
                            config=config
                        )
                        await process_response_text(response, message, bot_message, message_chunks)
                    # WEB SEARCH
                    elif fn.name == "browser":
                        q = fn.args.get('q', '')
                        num = fn.args.get('num', 15)
                        await bot_message.edit(content=f'-# Searching \'{q}\' <a:searchingweb:1246248294322147489>')
                        wsearch_result = await browser(q, num)
                        await bot_message.edit(content='-# Reading results... <a:searchingweb:1246248294322147489>')
                        function_response_part = types.Part.from_function_response(
                            name="browser",
                            response={"result": f"USE_CITATION=YES\nONLINE_RESULTS={wsearch_result}"}
                        )
                        chat_contents.append(types.Content(
                            role="model",
                            parts=[types.Part(function_call=fn)]
                        ))
                        chat_contents.append(types.Content(
                            role="user",
                            parts=[function_response_part]
                        ))
                        response = client.models.generate_content(
                            model=model_id,
                            contents=chat_contents,
                            config=config
                        )
                        await process_response_text(response, message, bot_message, message_chunks)
                    # IMAGINE
                    elif fn.name == "imagine":
                        prompt = fn.args.get('prompt', '')
                        ar = fn.args.get('ar', '1:1')
                        await bot_message.edit(content="-# Generating Image... <a:gemini_sparkles:1321895555676504077>")
                        imagine_result = await imagine(prompt, ar)
                        if imagine_result["is_error"] == 1:
                            await bot_message.edit(content='-# An Error Occurred <:error_icon:1295348741058068631>')
                            function_response_part = types.Part.from_function_response(
                                name="imagine",
                                response={"result": f"IMAGE_GENERATED=NO\nERROR_MSG=Error occurred on image model: {imagine_result['img_error_msg']}"}
                            )
                        else:
                            await bot_message.edit(content="-# Done <:checkmark:1220809843414270102>")
                            await message.reply(file=discord.File(imagine_result["filename"]))
                            os.remove(imagine_result["filename"])
                            function_response_part = types.Part.from_function_response(
                                name="imagine",
                                response={"result": "IMAGE_GENERATED=YES"}
                            )
                        chat_contents.append(types.Content(
                            role="model",
                            parts=[types.Part(function_call=fn)]
                        ))
                        chat_contents.append(types.Content(
                            role="user",
                            parts=[function_response_part]
                        ))
                        response = client.models.generate_content(
                            model=model_id,
                            contents=chat_contents,
                            config=config
                        )
                        await process_response_text(response, message, bot_message, message_chunks)
                elif chunk.text:
                    full_response += chunk.text
                    new_chunks = split_msg(full_response)
                    new_chunks[0] = new_chunks[0].replace("Gemini:", "", 1)
                    new_chunks[0] = new_chunks[0].replace("Language Model#3241:", "", 1)
                    
                    new_chunks = ["‎ " if chunk == "\n" else chunk for chunk in new_chunks]
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
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        # Finalize all chunks by removing the animation
        if message_chunks:
            for i, msg in enumerate(message_chunks):
                try:
                    await msg.edit(content=new_chunks[i])
                except Exception as e:
                    print(f"Error finalizing message {i}: {e}")
    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        print(f'Error handling message: {e}')
        if bot_message:
            await bot_message.edit(content=f'An error occurred: `{e}`')
        await asyncio.sleep(6)
        await bot_message.delete()

# Start the bot
try:
    bot.run(bot_token)
except Exception as e:
    print(f'Error starting the bot: {e}')
# lol
