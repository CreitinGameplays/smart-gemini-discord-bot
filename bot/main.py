import discord
import aiofiles
from discord.ext import commands
import asyncio
import os
import sys
from collections import defaultdict, deque
import datetime
import json
from json.decoder import JSONDecodeError
import re
import time
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
import traceback
import mimetypes
from pydub import AudioSegment
# MongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# Gemini SDK (new)
from google.genai import Client, types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# import everything from tools.py and config.py
from tools import *
from jupyter_manager import setup_jupyter, get_jupyter_manager
from config import return_system_prompt

# mongodb will be useful here
MONGO_URI = os.getenv('MONGO_URI')
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["gemini-bot-db"]

# Logging
handler = RotatingFileHandler(
    filename='bot_errors.log', mode='a', maxBytes=80 * 1024, backupCount=1, encoding='utf-8',
)
console_handler = logging.StreamHandler()  # Console output

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)  # Log errors only
logger.addHandler(handler)
logger.addHandler(console_handler)

# ENV
bot_token = os.getenv('TOKEN')
ai_key = os.getenv('GEMINI_KEY')

SEARCH_SNIPPET_SIZE = 6000
MAX_CHAT_HISTORY_MESSAGES = 24
allowed_ids = [775678427511783434] # creitin id xd

image_model_id = "imagen-3.0-fast-generate-001"
# Maintain last 10 attachments per type and per channel
attachment_histories = defaultdict(lambda: {
    "image": deque(maxlen=10),
    "audio": deque(maxlen=10),
    "text": deque(maxlen=10)
})

# SYSTEM PROMPT
base_system_prompt = str(return_system_prompt())

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
        "description": "Execute Python code snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "code_text": {"type": "string", "description": "Write the Python code here."}
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
#intents.message_content = True # supposed to still work without message intents, lets see

bot = discord.AutoShardedBot(intents=intents, shard_count=2)

# add cogs
cogs_list = [
    'settings',
    'help',
    'misc'
]
for cog in cogs_list:
    bot.load_extension(f'cogs.{cog}')
    
# classes for views
class PythonResultView(discord.ui.View):
    def __init__(self, result):
        super().__init__(timeout=None)
        self.result = result
    @discord.ui.button(label="Show Code", style=discord.ButtonStyle.grey, emoji="⚙️")
    async def button_callback(self, button, interaction):
        code_embed = discord.Embed(
            title="Python Code",
            description=f"```python\n{self.result}\n```",
            color=discord.Colour.blue()
        )
        code_embed.set_thumbnail(url="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png")
        await interaction.response.send_message(embed=code_embed, ephemeral=True)

class WebSearchResultView(discord.ui.View):
    def __init__(self, results):
        super().__init__(timeout=None)
        self.results = results
    @discord.ui.button(label="Sources", style=discord.ButtonStyle.grey, emoji="🌐")
    async def show_websites(self, button: discord.ui.Button, interaction: discord.Interaction):
        import re
        urls = re.findall(r'Link:\s*(\S+)', self.results)
        output = "\n".join(urls)
        if len(output) > 4090:
            output = output[:4000] + "..."
        sources_embed = discord.Embed(
            title="Sources",
            description=f"{output}",
            color=discord.Colour.nitro_pink()
        )
        sources_embed.set_thumbnail(url="https://cdn.revoltusercontent.com/attachments/_3Mg5mRzKc8fLNyxAjzoTqdPzdB2HS4molUCX75Fh2/pngegg.png")
        await interaction.response.send_message(embed=sources_embed, ephemeral=True)

def extract_youtube_url(text):
    if not text:
        return None

    youtube_regex = (
        r'(?i)(?:https?:\/\/)?(?:www\.)?(?:'
        r'youtube\.com\/(?:(?:watch\?(?:.*&)?v=)|(?:embed\/)|(?:v\/))|'
        r'youtu\.be\/'
        r')([a-zA-Z0-9_-]{11})(?:\S+)?'
    )
    match = re.search(youtube_regex, text)
    if match:
        video_id = match.group(1)
        # Return a normalized URL using the standard YouTube URL format
        return f"https://youtube.com/watch?v={video_id}"
    return None
    
# Restart
async def restart_bot():
    os.execv(sys.executable, ['python'] + sys.argv)

@bot.slash_command(name="install_lib", description="Install a Python library in the Jupyter environment (owner only).")
@commands.is_owner()
async def install_lib(ctx: discord.ApplicationContext, library: str):
    """Installs a library in the bot's Jupyter environment."""
    await ctx.defer(ephemeral=True)
    jupyter_manager = get_jupyter_manager()
    result = await jupyter_manager.install_library(library)
    if len(result) > 1900:
        result = result[:1900] + "..."
    await ctx.respond(f"Installation result for `{library}`:\n{result}", ephemeral=True)
    
async def check_response_timeout(bot_message, timeout=60, check_interval=5):
    initial_content = bot_message.content
    elapsed = 0

    while elapsed < timeout:
        await asyncio.sleep(check_interval)
        elapsed += check_interval
        try:
            # Refetch the latest version of the message.
            latest_message = await bot_message.channel.fetch_message(bot_message.id)
        except Exception as e:
            print(f"Error refetching message: {e}")
            break
        if latest_message.content != initial_content:
            return

    try:
        # If we get here, the message has not been updated over the timeout period.
        latest_message = await bot_message.channel.fetch_message(bot_message.id)
        await latest_message.edit(
            content=f"<:aw_snap:1379058439963017226> Sorry, the API did not return updated data for over {timeout} seconds. Please try again."
        )
        await asyncio.sleep(8)
        await latest_message.delete()
    except Exception as e:
        print(f"Timeout handling failed: {e}")

# Discord events
@bot.event
async def on_ready():
    if bot.auto_sync_commands:
        await bot.sync_commands() # since we want updated slash commands
    msg = discord.Game("Back to life! Made by Creitin Gameplays! 🌟")
    await bot.change_presence(status=discord.Status.online, activity=msg)
    print(f'Logged in as {bot.user}!')

@bot.event
async def on_application_command_error(ctx: discord.ApplicationContext, error: discord.DiscordException):
    """Global error handler for application commands."""
    if isinstance(error, commands.NotOwner):
        await ctx.respond("You are not authorized to use this command.", ephemeral=True)
    else:
        # Log the error for debugging
        logger.error(f"An error occurred in command '{ctx.command.qualified_name}':\n{traceback.format_exc()}")
        await ctx.respond(f"<:error_icon:1295348741058068631> An unexpected error occurred. Please check the logs.", ephemeral=True)

@bot.event
async def on_message(message):
    if message.guild is None:
        return
    # DATABASE SETTINGS
    server_settings = db.bot_settings.find_one({"server_id": message.guild.id})
    allowed_channels = server_settings.get("channels", []) if server_settings else []

    if message.author == bot.user:
        return

    if bot.user in message.mentions or (message.reference and message.reference.resolved.author == bot.user):
        if message.channel.id in allowed_channels:
            await handle_message(message)
        else:
            return

async def handle_message(message):
    # simple logging
    log_channel_id = bot.get_channel(1221244563407114240)
    await log_channel_id.send("```someone texted me!```")

    user_settings = None
    ### USER DATABASE SETTINGS (default values) ###
    temperature_setting = 0.6
    model_id = "gemini-2.5-flash"
    mention_author = True

    user_settings = db.bot_settings.find_one({"user_id": message.author.id})
    bot_message = None
    today2 = datetime.datetime.now()
    todayday2 = f'{today2.strftime("%A")}, {today2.month}/{today2.day}/{today2.year}'
    try:
        # database stuff
        if not user_settings:
            default_settings = {
                "temperature": 0.6,
                "model": model_id,
                "mention_author": True,
                "is_donator": False
            }
            db.bot_settings.update_one(
                {"user_id": message.author.id},
                {"$setOnInsert": default_settings},
                upsert=True
            )
            user_settings = default_settings
        else: # get the info
            model_id = user_settings.get("model", model_id)
            temperature_setting = user_settings.get("temperature", 0.6)
            mention_author = bool(user_settings.get("mention_author", True))

        channel_id = message.channel.id
        channel_history = [msg async for msg in message.channel.history(limit=MAX_CHAT_HISTORY_MESSAGES)]
        channel_history.reverse()

        chat_history = '\n'.join([f'{author}: {content}' for author, content in channel_histories[channel_id]])
        chat_history = chat_history.replace(f'<@{bot.user.id}>', '').strip()

        async with message.channel.typing():
            await asyncio.sleep(1)
            bot_message = await message.reply('<a:gemini_sparkles:1321895555676504077> _ _', mention_author=mention_author)
            asyncio.create_task(check_response_timeout(bot_message))
            await asyncio.sleep(0.1)
        user_message = message.content.replace(f'<@{bot.user.id}>', '').strip()

        attachment_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attachments')
        os.makedirs(attachment_folder, exist_ok=True)
        client = Client(api_key=ai_key)

        # --- AUDIO MIME AND EXTENSION SUPPORT ---
        AUDIO_MIME_MAP = {
            '.ogg':  'audio/ogg',
            '.mp3':  'audio/mpeg',
            '.wav':  'audio/wav',
            '.m4a':  'audio/mp4',
            '.flac': 'audio/flac',
            '.aac':  'audio/aac',
            '.opus': 'audio/opus',
            '.webm': 'audio/webm',
        }
        SUPPORTED_AUDIO_EXTS = tuple(AUDIO_MIME_MAP.keys())
        SUPPORTED_TEXT_EXTS = (
            '.txt', '.md', '.py', '.json', '.js', '.html', '.css', '.csv', '.yaml', '.yml', '.xml',
            '.c', '.cpp', '.java', '.ts', '.sh', '.bat', '.ini', '.conf', '.toml', '.log'
        )

        def classify_attachment(attachment):
            filename = attachment.filename.lower()
            if attachment.content_type and attachment.content_type.startswith('image'):
                return 'image'
            if attachment.content_type and attachment.content_type.startswith('audio'):
                return 'audio'
            if filename.endswith(SUPPORTED_AUDIO_EXTS):
                return 'audio'
            if (attachment.content_type and attachment.content_type.startswith('text')) or any(filename.endswith(ext) for ext in SUPPORTED_TEXT_EXTS):
                return 'text'
            return None

        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="AUTO"
            )
        )
        ### test 
        if model_id != "gemini-2.0-flash":
            config = types.GenerateContentConfig(
                temperature=temperature_setting,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
                tools=[tool_python, tool_websearch, tool_imagine],
                tool_config=tool_config,
                system_instruction=[
                    types.Part.from_text(text=base_system_prompt.replace("TODAYTIME00", todayday2).replace("GEMINIMODELID", model_id))
                ],
                thinking_config = types.ThinkingConfig(
                    thinking_budget=-1, # dynamic thinking according to https://ai.google.dev/gemini-api/docs/thinking#set-budget
                    include_thoughts=False
                ),
            )
        else:
            config = types.GenerateContentConfig(
                temperature=temperature_setting,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
                tools=[tool_python, tool_websearch, tool_imagine],
                tool_config=tool_config,
                system_instruction=[
                    types.Part.from_text(text=base_system_prompt.replace("TODAYTIME00", todayday2).replace("GEMINIMODELID", model_id))
                ]
            )
        #### i hope this works :pray:
        chat_contents = []
        # Add chat history (text)
        for m in channel_history:
            author_name = str(m.author)
            if author_name == "Gemini Pro#6900":
                text = f"{m.content}"
            else:
                text = f"{author_name}: {m.content}"
            chat_contents.append(
                types.Content(
                    role="user" if m.author.name != bot.user.name else "model",
                    parts=[types.Part.from_text(text=text)]
                )
            )

        # 1. Reset attachment_histories for this channel
        attachment_histories[channel_id]["image"].clear()
        attachment_histories[channel_id]["audio"].clear()
        attachment_histories[channel_id]["text"].clear()

        # 2. Scan only last 10 messages for attachments and save them (plus any in current msg)
        for m in channel_history[-10:]:
            if m.attachments:
                for attachment in m.attachments:
                    atype = classify_attachment(attachment)
                    if atype:
                        fname = f"{m.id}_{attachment.filename}"
                        if atype == 'text':
                            base, _ = os.path.splitext(fname)
                            fname = base + ".txt"
                        elif atype == 'audio':
                            base, ext = os.path.splitext(fname)
                            ext = ext if ext in AUDIO_MIME_MAP else '.ogg'
                            fname = base + ext
                        fpath = os.path.join(attachment_folder, fname)
                        if not os.path.exists(fpath):
                            data = await attachment.read()
                            async with aiofiles.open(fpath, 'wb') as f:
                                await f.write(data)
                        if fpath not in attachment_histories[channel_id][atype]:
                            attachment_histories[channel_id][atype].append(fpath)

        # 3. Handle new attachments in the current message
        if message.attachments:
            for attachment in message.attachments:
                atype = classify_attachment(attachment)
                if atype:
                    fname = f"{message.id}_{attachment.filename}"
                    if atype == 'text':
                        base, _ = os.path.splitext(fname)
                        fname = base + ".txt"
                    elif atype == 'audio':
                        base, ext = os.path.splitext(fname)
                        ext = ext if ext in AUDIO_MIME_MAP else '.ogg'
                        fname = base + ext
                    fpath = os.path.join(attachment_folder, fname)
                    data = await attachment.read()
                    async with aiofiles.open(fpath, 'wb') as f:
                        await f.write(data)
                    if fpath not in attachment_histories[channel_id][atype]:
                        attachment_histories[channel_id][atype].append(fpath)

        # 4. Remove files not referenced in last 10 messages or this message
        valid_files = set()
        for atype in ['image', 'audio', 'text']:
            valid_files.update(attachment_histories[channel_id][atype])
        for fname in os.listdir(attachment_folder):
            fpath = os.path.join(attachment_folder, fname)
            if fpath not in valid_files:
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        # 5. Upload attachments to Gemini and add to chat_contents
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

        for atype, default_mime, instr in [
            ('image', 'image/png', '[Instructions: This is an image. Use as context.]'),
            ('audio', None, '[Instructions: This is an audio file. Use as context.]'),
            ('text', 'text/plain', '[Instructions: This is a text file. Use as context.]')
        ]:
            for fpath in list(attachment_histories[channel_id][atype])[-3:]:
                if os.path.exists(fpath):
                    if atype == 'audio':
                        file_ext = os.path.splitext(fpath)[1].lower()
                        if file_ext == ".ogg":
                            mp3_path = os.path.splitext(fpath)[0] + ".mp3"
                            if not os.path.exists(mp3_path):
                                audio = AudioSegment.from_ogg(fpath)
                                audio.export(mp3_path, format="mp3")
                            fpath = mp3_path
                            mime = 'audio/mpeg'
                        else:
                            mime_type, _ = mimetypes.guess_type(fpath)
                            if not mime_type:
                                ext = os.path.splitext(fpath)[1].lower()
                                mime_type = AUDIO_MIME_MAP.get(ext, 'audio/mpeg')
                            mime = mime_type
                    else:
                        mime = default_mime
                    uploaded = await upload_to_gemini(fpath)
                    chat_contents.append(types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime),
                            types.Part.from_text(text=instr)
                        ]
                    ))

        youtube_url = extract_youtube_url(message.content)
        if youtube_url:
            chat_contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=youtube_url,
                            mime_type="video/*"
                        ),
                        types.Part.from_text(
                            text="[Instructions: Process this YouTube video and respond to the user about its content.]"
                        )
                    ]
                )
            )
            user_message += " [This message contains a YouTube video...]"
            print(f"Processing YouTube URL: {youtube_url}")

        chat_contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        ))
        # bug: this is synchronous, so this will block other bot interaction commands
        """
        response_stream = client.models.generate_content_stream(
            model=model_id,
            contents=chat_contents,
            config=config
        )
        """

        loop = asyncio.get_running_loop()
        response_stream = await loop.run_in_executor(
            None,
            lambda: list(client.models.generate_content_stream(
                model=model_id,
                contents=chat_contents,
                config=config
            ))
        )
        
        full_response = ""
        message_chunks = []
        aggregated_wsearch_results = ""
        view_for_continued_message = None

        # Process Gemini response
        while True:
            for chunk in response_stream:
                try:
                    # If text is included, accumulate and update messages.
                    if chunk.text:
                        full_response += chunk.text
                        new_chunks = split_msg(full_response)

                        for i in range(len(new_chunks)):
                            current_view = view_for_continued_message if i == 0 and view_for_continued_message else discord.utils.MISSING
                            if i < len(message_chunks):
                                await message_chunks[i].edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>", view=current_view)
                                await asyncio.sleep(0.8)
                            else:
                                if i == 0:
                                    await bot_message.edit(content=new_chunks[i] + " <a:generatingslow:1246630905632653373>", view=current_view)
                                    await asyncio.sleep(0.8)
                                    message_chunks.append(bot_message)
                                else:
                                    new_msg = await message.reply(new_chunks[i] + " <a:generatingslow:1246630905632653373>", mention_author=mention_author)
                                    await asyncio.sleep(0.8)
                                    message_chunks.append(new_msg)

                    if chunk.function_calls:
                        fn = chunk.function_calls[0]
                        current_response = full_response  # Save text before the tool call
                        print(fn) # debug

                        # Determine which message to edit for tool status updates
                        tool_update_message = bot_message
                        was_mid_stream = full_response.strip()

                        if was_mid_stream:
                            if message_chunks:
                                # Finalize the last message chunk by removing the thinking animation
                                last_text_chunk_content = split_msg(full_response)[-1]
                                await message_chunks[-1].edit(content=last_text_chunk_content)
                            # Create a new, temporary message for tool status
                            tool_update_message = await message.reply(content="-# <a:gemini_sparkles:1321895555676504077>", mention_author=mention_author)

                        if fn.name == "python":
                            code_text = fn.args.get('code_text', '')
                            await tool_update_message.edit(content=f"-# Executing... <a:brackets:1300121114869235752>")
                            jupyter_manager = get_jupyter_manager()
                            python_result = await jupyter_manager.execute_code(code_text)
                            python_view = PythonResultView(result=code_text)
                            await tool_update_message.edit(content=f"-# Done <:checkmark0:1246546819710849144>", view=python_view)
                            view_for_continued_message = python_view
                            function_response_part = types.Part.from_function_response(
                                name="python",
                                response={"result": python_result}
                            )
                        elif fn.name == "browser":
                            q = fn.args.get('q', '')
                            num = fn.args.get('num', 15)
                            await tool_update_message.edit(content=f'-# Searching \'{q}\' <a:searchingweb:1246248294322147489>')
                            wsearch_result = await browser(q, num)
                            aggregated_wsearch_results += wsearch_result
                            web_view = WebSearchResultView(results=aggregated_wsearch_results)
                            await tool_update_message.edit(content='-# Reading results... <a:searchingweb:1246248294322147489>', view=web_view)
                            view_for_continued_message = web_view
                            function_response_part = types.Part.from_function_response(
                                name="browser",
                                response={"result": f"USE_CITATION=YES\nONLINE_RESULTS={wsearch_result}"}
                            )
                        elif fn.name == "imagine":
                            prompt = fn.args.get('prompt', '')
                            ar = fn.args.get('ar', '1:1')
                            await tool_update_message.edit(content="-# Generating Image... <a:gemini_sparkles:1321895555676504077>")
                            imagine_result = await imagine(prompt, ar, message.author.id)
                            if imagine_result["is_error"] == 1:
                                await tool_update_message.edit(content='-# An Error Occurred <:error_icon:1295348741058068631>')
                                function_response_part = types.Part.from_function_response(
                                    name="imagine",
                                    response={"result": f"IMAGE_GENERATED=NO\nERROR_MSG=Error occurred: {imagine_result['img_error_msg']}"}
                                )
                            else:
                                await message.reply(file=discord.File(imagine_result["filename"]), mention_author=mention_author)
                                os.remove(imagine_result["filename"])
                                function_response_part = types.Part.from_function_response(
                                    name="imagine",
                                    response={"result": "IMAGE_GENERATED=YES"}
                                )
                                # Clean up the status message
                                if tool_update_message != bot_message:
                                    await tool_update_message.delete()
                                else:
                                    await tool_update_message.edit(content="-# Done <:checkmark:1220809843414270102>")
                        # Append the function call and its result to the history.
                        chat_contents.append(types.Content(
                            role="model",
                            parts=[types.Part(function_call=fn)]
                        ))
                        chat_contents.append(types.Content(
                            role="user",
                            parts=[function_response_part]
                        ))
                        # Restart the stream with updated chat_contents.
                        response_stream = client.models.generate_content_stream(
                            model=model_id,
                            contents=chat_contents,
                            config=config
                        )
                        # Set up state for the next iteration of the while loop.
                        if was_mid_stream and fn.name in ["python", "browser"]:
                            # The tool call created a new message that we want to continue generating from. To remove the tool status text, we reset the response content.
                            fetched_tool_message = await message.channel.fetch_message(tool_update_message.id) # We still need the message object
                            full_response = ""
                            message_chunks = [fetched_tool_message]
                        else:
                            # Start fresh if tool was first action, or resume normally for imagine
                            full_response = current_response if was_mid_stream else ""
                            view_for_continued_message = None
                        break
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping invalid JSON chunk: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
            else:
                break

        # Finalize all messages by removing the animation icon.
        if message_chunks:
            for i, msg in enumerate(message_chunks):
                try:
                    final_view = view_for_continued_message if i == 0 and view_for_continued_message else discord.utils.MISSING
                    await msg.edit(content=split_msg(full_response)[i], view=final_view)
                except Exception as e:
                    print(f"Error finalizing message {i}: {e}")

    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        print(f'Error handling message: {e}')
        if bot_message:
            await bot_message.edit(content=f'<:error_icon:1295348741058068631> An error occurred: `{e}`')
        await asyncio.sleep(6)
        await bot_message.delete()

# Start the bot
try:
    setup_jupyter()
    bot.run(bot_token)
except Exception as e:
    print(f'Error starting the bot: {e}')
# lol
