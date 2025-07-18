import discord
from discord.ext import commands
from discord import option
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import imagine

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import base64
from google import genai
from google.genai import types
import re
import asyncio

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["gemini-bot-db"]

# Split message function
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

# A view with a red delete button that can remove the image message.
class DeleteView(discord.ui.View):
    def __init__(self, author_id: int):
        super().__init__(timeout=None)
        self.author_id = author_id

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.danger, emoji="❌")
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(":x: Only the command author can delete this image.", ephemeral=True)
        else:
            await interaction.message.delete()
            await interaction.response.send_message("**Image deleted!**", ephemeral=True)

class ModelInfoView(discord.ui.View):
    def __init__(self, model: str):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label=f"Model: {model}", style=discord.ButtonStyle.secondary, disabled=True))
        
class Misc(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @discord.slash_command(
        name="imagine", 
        description="Generate an image using Imagen 3 [Donators only]",
        integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        },
    )
    @option(name="prompt", description="the prompt that will be used for image generation")
    async def imagine(
        self, 
        ctx: discord.ApplicationContext, 
        prompt: str, 
        ar: str = discord.Option(
            choices=["16:9", "9:16", "1:1"], 
            description="Select the image aspect ratio."
        )
    ):
        # Check if the user is a donator
        user_settings = db.bot_settings.find_one({"user_id": ctx.author.id})
        if not user_settings or not user_settings.get("is_donator", False):
            await ctx.respond("<:info:1220157026739552296> You must be a donator to use the `/imagine` command.", ephemeral=True)
            return

        await ctx.defer()
        result = await imagine(prompt, ar, ctx.author.id)
        if result.get("is_error", 1) == 1:
            await ctx.respond(f"Error generating image: {result.get('img_error_msg', 'Unknown error')}", ephemeral=True)
        else:
            filename = result.get("filename")
            file = discord.File(filename, filename=filename)
            view = DeleteView(ctx.author.id)
            await ctx.respond("<:checkmark0:1246546819710849144> Here is your generated image:", file=file, view=view)


    @discord.slash_command(
        name="ask", 
        description="Ask a question using Gemini AI.",
        integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        },
    )
    @option(name="prompt", description="The question or prompt to ask")
    async def ask(self, ctx: discord.ApplicationContext, prompt: str):
        # logging
        log_channel_id = self.bot.get_channel(1221244563407114240)
        await log_channel_id.send("```someone texted me with /ask!```")

        await ctx.defer()
        # Retrieve user's model from settings or use default.
        user_settings = db.bot_settings.find_one({"user_id": ctx.author.id})
        model = user_settings.get("model") if user_settings and user_settings.get("model") else "gemini-2.5-flash"
        temperature = user_settings.get("temperature") if user_settings and user_settings.get("temperature") else 0.6
        try:
            client = genai.Client(api_key=os.environ.get("GEMINI_KEY"))
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
            if model != "gemini-2.0-flash":
                generate_config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="text/plain",
                    thinking_config = types.ThinkingConfig(
                        thinking_budget=-1, # dynamic thinking according to https://ai.google.dev/gemini-api/docs/thinking#set-budget
                        include_thoughts=False
                    ),
                )
            else:
                generate_config = types.GenerateContentConfig(temperature=temperature, response_mime_type="text/plain")
            response_text = ""
            await ctx.respond("<a:gemini_sparkles:1321895555676504077> _ _")
            
            # Stream generation in a blocking call wrapped in run_in_executor.
            def stream_generation():
                return list(client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_config,
                ))
            chunks = await ctx.bot.loop.run_in_executor(None, stream_generation)
            
            # Accumulate streaming response.
            for chunk in chunks:
                text = chunk.text if hasattr(chunk, "text") else (chunk[0] if isinstance(chunk, (list, tuple)) else "")
                response_text += text
                if len(response_text) < 1950:
                    await ctx.edit(content=response_text)
                await asyncio.sleep(0.1)
                
            # Now split the complete response if it exceeds Discord's limit.
            response_chunks = split_msg(response_text, chunk_size=1500)
            view = ModelInfoView(model)
            await ctx.edit(content=response_chunks[0], view=view)
            for chunk in response_chunks[1:]:
                await ctx.followup.send(chunk)
        except Exception as e:
            await ctx.respond(f"<:error_icon:1295348741058068631> An error occurred: {e}", ephemeral=True)

def setup(bot: commands.Bot):
    bot.add_cog(Misc(bot))
