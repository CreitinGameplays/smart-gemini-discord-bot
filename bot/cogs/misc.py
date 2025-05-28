import discord
from discord.ext import commands
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import imagine

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["gemini-bot-db"]

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

class Misc(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @discord.slash_command(
        name="imagine", 
        description="Generate an image using Imagen 3 [Donators only]"
    )
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

        await ctx.defer(ephemeral=True)
        result = await imagine(prompt, ar, ctx.author.id)
        if result.get("is_error", 1) == 1:
            await ctx.respond(f"Error generating image: {result.get('img_error_msg', 'Unknown error')}", ephemeral=True)
        else:
            filename = result.get("filename")
            file = discord.File(filename, filename=filename)
            view = DeleteView(ctx.author.id)
            await ctx.respond("<:checkmark0:1246546819710849144> Here is your generated image:", file=file, view=view, ephemeral=False)

def setup(bot: commands.Bot):
    bot.add_cog(Misc(bot))