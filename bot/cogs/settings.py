import discord
from discord.ext import commands
from typing import Union
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["gemini-bot-db"]

temp_description = "**`temperature` controls the model randomness**: higher values yield more creative responses, while lower values produce more focused and deterministic results."
model_description = "Choose a Gemini model that fits your needs: the high-performance pro version for donators or the flash versions for reliable standard responses."

# Modal for updating the temperature.
class TemperatureModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="Set Temperature")
        self.title = "Set Temperature"
        self.input_temperature = discord.ui.InputText(
            label="Temperature (0-2)",
            placeholder="Enter a float value between 0 and 2"
        )
        self.add_item(self.input_temperature)
    
    async def callback(self, interaction: discord.Interaction):
        try:
            value = float(self.input_temperature.value)
            if not (0 <= value <= 2):
                await interaction.response.send_message("<:alert:1220162599014895777> Temperature must be a **float value between 0 and 2**.", ephemeral=True)
                return
            db.bot_settings.update_one(
                {"user_id": interaction.user.id},
                {"$set": {"temperature": value}},
                upsert=True
            )
            await interaction.response.send_message(f"<a:verificadoTESTE:799380003426795561> Temperature set to {value}.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"<:error_icon:1295348741058068631> Error setting temperature: {e}", ephemeral=True)

# Dropdown for selecting Gemini model.
class ModelDropdown(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="gemini-2.5-pro", description="Donator only", emoji="🔥"),
            discord.SelectOption(label="gemini-2.5-pro-preview-05-06", description="Donator only", emoji="🔥"),
            discord.SelectOption(label="gemini-2.5-pro-preview-06-05", description="Donator only", emoji="🔥"),
            discord.SelectOption(label="gemini-2.5-flash", description="Standard model", emoji="<:gemini:1219726550195245127>"),
            discord.SelectOption(label="gemini-2.5-flash-preview-05-20", description="Standard model", emoji="<:gemini:1219726550195245127>"),
            discord.SelectOption(label="gemini-2.5-flash-preview-04-17", description="Standard model", emoji="<:gemini:1219726550195245127>"),
            discord.SelectOption(label="gemini-2.0-flash", description="Older model", emoji="<:gemini:1219726550195245127>"),
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
        # Check if the selected model is restricted for donators.
        if (model == "gemini-2.5-pro-preview-05-06" or model == "gemini-2.5-pro-preview-06-05" or model == "gemini-2.5-pro") and not user_settings.get("is_donator", False):
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
    
    @discord.ui.button(label="Set Temperature", style=discord.ButtonStyle.grey, emoji="🌡")
    async def temperature_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        modal = TemperatureModal()
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Reload", style=discord.ButtonStyle.grey, emoji="<:reload:1377257463317008466>")
    async def reload_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        user_settings = db.bot_settings.find_one({"user_id": interaction.user.id}) or {}
        temperature = user_settings.get("temperature", 0.6)
        model = user_settings.get("model", "gemini-2.5-flash-preview-05-20")
        mention = user_settings.get("mention_author", True)
        
        new_embed = discord.Embed(
            title="Your Bot Settings",
            description="Use the interactive menu below to update your settings. Your current setup:",
            color=discord.Colour.blue()
        )
        new_embed.add_field(name="Temperature", value=f"`{temperature}`", inline=True)
        new_embed.add_field(name="Gemini Model", value=f"`{model}`", inline=True)
        new_embed.add_field(name="Mention", value=f"`{'Yes' if mention else 'No'}`", inline=True)
        new_embed.add_field(name="Temperature Description", value=temp_description, inline=False)
        new_embed.add_field(name="Models", value=model_description, inline=False)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_embed.set_footer(text=f"Requested by {interaction.user.name} on {current_time}", icon_url=interaction.user.avatar.url)
        if interaction.client.user and interaction.client.user.avatar:
            new_embed.set_thumbnail(url=interaction.client.user.avatar.url)
            
        await interaction.response.edit_message(embed=new_embed)

# Settings cog that consolidates all settings modifications into one command.
class Settings(commands.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot = bot
    
    @discord.slash_command(name="settings", description="Display and adjust your bot settings.", integration_types={discord.IntegrationType.guild_install, discord.IntegrationType.user_install})
    async def settings(self, ctx: discord.ApplicationContext):
        try:
            user_settings = db.bot_settings.find_one({"user_id": ctx.author.id}) or {}
            temperature = user_settings.get("temperature", 0.6)
            model = user_settings.get("model", "gemini-2.5-flash")
            mention = user_settings.get("mention_author", True)
            
            embed = discord.Embed(
                title="Your Bot Settings",
                description="Use the interactive menu below to update your settings. Your current setup:",
                color=discord.Colour.blue()
            )
            embed.add_field(name="Temperature", value=f"`{temperature}`", inline=True)
            embed.add_field(name="Gemini Model", value=f"`{model}`", inline=True)
            embed.add_field(name="Mention", value=f"`{'Yes' if mention else 'No'}`", inline=True)
            embed.add_field(name="Temperature Description", value=temp_description, inline=False)
            embed.add_field(name="Models", value=model_description, inline=False)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            embed.set_footer(text=f"Requested by {ctx.author.name} on {current_time}", icon_url=ctx.user.avatar.url)
            if self.bot.user and self.bot.user.avatar:
                embed.set_thumbnail(url=self.bot.user.avatar.url)
            
            view = SettingsView(self.bot)
            await ctx.respond(embed=embed, view=view, ephemeral=True)
        except Exception as e:
            await ctx.respond(f"<:error_icon:1295348741058068631> An error occurred while retrieving your settings: {e}", ephemeral=True)
            print(f"Error in /settings command: {e}")

    channel = discord.SlashCommandGroup("channel", "Manage channels where the bot is allowed to respond.")

    @channel.command(name="add", description="Add a channel where the bot can respond.")
    @commands.has_permissions(manage_guild=True)
    async def add_channel(self, ctx: discord.ApplicationContext, channel: discord.TextChannel):
        try:
            result = db.bot_settings.update_one(
                {"server_id": ctx.guild.id},
                {"$addToSet": {"channels": channel.id}},
                upsert=True
            )
            if result.modified_count == 0 and result.upserted_id is None:
                await ctx.respond("<:alert:1220162599014895777> This channel has already been added.", ephemeral=True)
            else:
                await ctx.respond(f"<a:verificadoTESTE:799380003426795561> <#{channel.id}> has been **added!**", ephemeral=True)
        except Exception as e:
            await ctx.respond(f"<:error_icon:1295348741058068631> An error occurred while adding the channel: {e}", ephemeral=True)
            print(f"Error in channel add command: {e}")

    @add_channel.error
    async def add_channel_error(self, ctx: discord.ApplicationContext, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.respond("<:alert:1220162599014895777> You do not have the required permissions to add channels (Manage Guild permission required).", ephemeral=True)
        else:
            await ctx.respond(f"<:error_icon:1295348741058068631> An unexpected error occurred: {error}", ephemeral=True)

    @channel.command(name="remove", description="Remove a channel from the allowed list.")
    @commands.has_permissions(manage_guild=True)
    async def remove_channel(self, ctx: discord.ApplicationContext, channel: discord.TextChannel):
        try:
            result = db.bot_settings.update_one(
                {"server_id": ctx.guild.id},
                {"$pull": {"channels": channel.id}}
            )
            if result.modified_count == 0:
                await ctx.respond("<:alert:1220162599014895777> This channel was not found in the list.", ephemeral=True)
            else:
                await ctx.respond(f"<a:verificadoTESTE:799380003426795561> <#{channel.id}> has been **removed!**", ephemeral=True)
        except Exception as e:
            await ctx.respond(f"<:error_icon:1295348741058068631> An error occurred while removing the channel: {e}", ephemeral=True)
            print(f"Error in channel remove command: {e}")

    @remove_channel.error
    async def remove_channel_error(self, ctx: discord.ApplicationContext, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.respond("<:alert:1220162599014895777> You do not have the required permissions to remove channels (Manage Guild permission required).", ephemeral=True)
        else:
            await ctx.respond(f"<:error_icon:1295348741058068631> An unexpected error occurred: {error}", ephemeral=True)
            
    @channel.command(name="list", description="List allowed channels in this server.")
    async def list_channels(self, ctx: discord.ApplicationContext):
        try:
            server_settings = db.bot_settings.find_one({"server_id": ctx.guild.id})
            channels = server_settings.get("channels", []) if server_settings else []
            if channels:
                channel_mentions = ", ".join(f"<#{cid}>" for cid in channels)
                description = channel_mentions
            else:
                description = "No channels have been added yet."
            embed = discord.Embed(
                title="List of Allowed Channels",
                description=description,
                color=discord.Colour.gold()
            )
            embed.set_thumbnail(url=self.bot.user.avatar.url)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            embed.set_footer(text=f"Requested by {ctx.author.name} on {current_time}")
            await ctx.respond(embed=embed)
        except Exception as e:
            await ctx.respond(f"<:error_icon:1295348741058068631> An error occurred while listing channels: {e}", ephemeral=True)
            print(f"Error in channel list command: {e}")

def setup(bot: discord.Bot):
    bot.add_cog(Settings(bot))
