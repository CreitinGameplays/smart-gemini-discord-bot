import discord
from discord.ext import commands
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["gemini-bot-db"]

# Modal for updating the temperature.
class TemperatureModal(discord.ui.Modal):
    temperature = discord.ui.InputText(
        label="Temperature (0-2)",
        placeholder="Enter a value between 0 and 2"
    )
    
    def __init__(self):
        super().__init__(title="Set Temperature")
    
    async def callback(self, interaction: discord.Interaction):
        try:
            value = float(self.temperature.value)
            if not (0 <= value <= 2):
                await interaction.response.send_message(":x: Temperature must be between 0 and 2.", ephemeral=True)
                return
            db.bot_settings.update_one(
                {"user_id": interaction.user.id},
                {"$set": {"temperature": value}},
                upsert=True
            )
            await interaction.response.send_message(f"<a:verificadoTESTE:799380003426795561> Temperature set to {value}.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f":x: Error setting temperature: {e}", ephemeral=True)

# Dropdown for selecting Gemini model.
class ModelDropdown(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="gemini-2.5-pro-preview-05-06", description="Donator only", emoji="🔥"),
            discord.SelectOption(label="gemini-2.5-flash-preview-05-20", description="Standard model"),
            discord.SelectOption(label="gemini-2.5-flash-preview-04-17", description="Standard model"),
            discord.SelectOption(label="gemini-2.0-flash", description="Older model"),
        ]
        super().__init__(
            placeholder="Select Gemini Model...",
            min_values=1,
            max_values=1,
            options=options
        )
    
    async def callback(self, interaction: discord.Interaction):
        model = self.values[0]
        user_settings = db.bot_settings.find_one({"user_id": interaction.user.id}) or {}
        # Check if the selected model is restricted.
        if model == "gemini-2.5-pro-preview-05-06" and not user_settings.get("is_donator", False):
            await interaction.response.send_message("<:info:1220157026739552296> This model is only available for donators.", ephemeral=True)
            return
        db.bot_settings.update_one(
            {"user_id": interaction.user.id},
            {"$set": {"model": model}},
            upsert=True
        )
        await interaction.response.send_message(f"<a:verificadoTESTE:799380003426795561> Gemini model set to `{model}`.", ephemeral=True)

# Dropdown for setting mention preference.
class MentionDropdown(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="Yes", description="Bot will mention you."),
            discord.SelectOption(label="No", description="Bot will not mention you."),
        ]
        super().__init__(
            placeholder="Select Mention Preference...",
            min_values=1,
            max_values=1,
            options=options
        )
    
    async def callback(self, interaction: discord.Interaction):
        selection = self.values[0]
        mention = True if selection == "Yes" else False
        db.bot_settings.update_one(
            {"user_id": interaction.user.id},
            {"$set": {"mention_author": mention}},
            upsert=True
        )
        msg = "<a:verificadoTESTE:799380003426795561> The bot will now mention you in responses." if mention else "<a:verificadoTESTE:799380003426795561> The bot will no longer mention you in responses."
        await interaction.response.send_message(msg, ephemeral=True)

# Main interactive view containing the dropdowns and a button to update the temperature.
class SettingsView(discord.ui.View):
    def __init__(self, bot: discord.Bot):
        super().__init__(timeout=180)
        self.bot = bot
        self.add_item(ModelDropdown())
        self.add_item(MentionDropdown())
    
    @discord.ui.button(label="Set Temperature", style=discord.ButtonStyle.primary, emoji="🌡")
    async def temperature_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        modal = TemperatureModal()
        await interaction.response.send_modal(modal)

# Settings cog that consolidates all settings modifications into one command.
class Settings(commands.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot = bot
    
    @discord.slash_command(name="settings", description="Display and adjust your bot settings.")
    async def settings(self, ctx: discord.ApplicationContext):
        try:
            user_settings = db.bot_settings.find_one({"user_id": ctx.author.id}) or {}
            temperature = user_settings.get("temperature", 0.6)
            model = user_settings.get("model", "gemini-2.5-flash-preview-05-20")
            mention = user_settings.get("mention_author", True)
            
            embed = discord.Embed(
                title="Your Bot Settings",
                description="Use the interactive menu below to update your settings.",
                color=discord.Colour.blue()
            )
            embed.add_field(name="Temperature", value=f"`{temperature}`", inline=True)
            embed.add_field(name="Gemini Model", value=f"`{model}`", inline=True)
            embed.add_field(name="Mention", value=f"`{'Yes' if mention else 'No'}`", inline=True)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            embed.set_footer(text=f"Requested by {ctx.author.name} on {current_time}")
            if self.bot.user and self.bot.user.avatar:
                embed.set_thumbnail(url=self.bot.user.avatar.url)
            
            view = SettingsView(self.bot)
            await ctx.respond(embed=embed, view=view, ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while retrieving your settings: {e}", ephemeral=True)
            print(f"Error in /settings command: {e}")

def setup(bot: discord.Bot):
    bot.add_cog(Settings(bot))