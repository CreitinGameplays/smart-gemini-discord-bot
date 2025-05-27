import discord
from discord.ext import commands
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from typing import Union
from discord import option
import datetime
from datetime import datetime

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
bot_owner_id = 775678427511783434 # creitin id xd
db = mongo_client["gemini-bot-db"]

async def setup_mongodb():
    try:
        # Check if the database exists
        c_list = db.list_collections()
        if 'bot_settings' not in c_list:
            print("Creating database...")
            db.create_collection("bot_settings")
            return "Database and collection created successfully."
        else:
            print("Database already exists.")
            return "Database already exists."
    except Exception as e:
        print(f"Error setting up MongoDB: {e}")
        return f"Error setting up MongoDB: {e}"

class Settings(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    """
    TODO: Add commands to change bot settings, etc.
    """
    @commands.Cog.listener()
    async def on_message(self, message):
        # owner only command obv
        if message.author.bot:
            return
        try:
            if message.content.startswith('.setupdb'):
                if message.author.id != bot_owner_id:
                    await message.reply(":x: You do not have permission to use this command.")
                    return
                else:
                    result = await setup_mongodb()
                    await message.reply(f"MongoDB setup status: {result}")
            
        except Exception as e:
            await message.reply(f":x: An error occurred: {e}")
            print(f"Error in on_message: {e}")

    # group slash commands
    settings = discord.SlashCommandGroup("settings", "Manage bot settings.") 
    set_temperature = settings.create_subgroup(
        "temperature",
        "Set the AI temperature value (0-2)."
    )
    channels_setting = settings.create_subgroup(
        "channel",
        "Manage channels that the bot should respond"
    )
    change_model = settings.create_subgroup(
        "model",
        "Change the AI model used by the bot."
    )

    @settings.command(name="open", description="Display your current bot settings and instructions to manage them.")
    async def open(self, ctx: discord.ApplicationContext):
        try:
            # Retrieve user settings from the database for this user, using defaults if none exist.
            user_settings = db.bot_settings.find_one({"user_id": ctx.author.id})
            if user_settings is None:
                # Defaults if settings not found.
                temperature = 0.6
                model = "gemini-2.5-flash-preview-05-20"
                mention_author = True
            else:
                temperature = user_settings.get("temperature", 0.6)
                model = user_settings.get("model", "gemini-2.5-flash-preview-05-20")
                mention_author = user_settings.get("mention_author", True)
            
            # Create an embed that displays the current settings.
            embed = discord.Embed(
                title="Current Bot Settings",
                description="Below are your current settings. Use the subcommands to update any setting.",
                color=discord.Colour.blue()
            )
            embed.add_field(name="Temperature", value=f"`{temperature}`", inline=True)
            embed.add_field(name="Gemini Model", value=f"`{model}`", inline=True)
            embed.add_field(name="Mention Preference", value=f"`{'Yes' if mention_author else 'No'}`", inline=True)
            embed.add_field(
                name="Next Steps",
                value=("To change any of these settings, use the following subcommands:\n"
                    "- `/settings temperature set [value]` to adjust the AI temperature. This **controls the model randomness**: higher values yield more creative responses, while lower values produce more focused and deterministic results.\n"
                    "- `/settings model set [model]` to change Gemini model.\n"
                    "- `/settings mention mention [True/False]` to set your mention preference.\n"
                    "- For managing allowed channels, use `/settings/channel add/remove/list`.\n"),
                inline=False
            )
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            embed.set_footer(text=f"Requested by {ctx.author.name} on {current_time}")
            embed.set_thumbnail(url=bot.user.avatar.url)
            
            await ctx.respond(embed=embed, ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while retrieving your settings: {e}", ephemeral=True)
            print(f"Error in settings open command: {e}")

    @set_temperature.command(description="Set the AI temperature value (0-2).")
    async def set(self, ctx: discord.ApplicationContext, value: float):
        if value < 0 or value > 2:
            await ctx.respond(":x: `temperature` value must be a float between 0 and 2.", ephemeral=True)
            return
        try:
            # Update the temperature in the database for this user id
            db.bot_settings.update_one(
                {"user_id": ctx.author.id},
                {"$set": {"temperature": value}},
                upsert=True
            )
            await ctx.respond(f"<a:verificadoTESTE:799380003426795561> Model `temperature` set to {value}.", ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while setting the temperature: {e}", ephemeral=True)
            print(f"Error in set_temperature: {e}")
    
    @channels_setting.command(description="Add one or more channels to the bot to respond")
    @commands.has_permissions(manage_guild=True)
    async def add(self, ctx: discord.ApplicationContext, channel: Union[discord.TextChannel]):
        try:
            result = db.bot_settings.update_one(
                {"server_id": ctx.guild_id},
                {"$addToSet": {"channels": channel.id}},
                upsert=True
            )
            # If modified_count is zero and no upsert happened, then the channel was already in the list.
            if result.modified_count == 0 and result.upserted_id is None:
                await ctx.respond(":warning: This channel has already been added.", ephemeral=True)
            else:
                await ctx.respond(f"<a:verificadoTESTE:799380003426795561> <#{channel.id}> has been **added!**", ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while adding the channel: {e}", ephemeral=True)
            print(f"Error in channels_setting: {e}")

    @add.error
    async def add_error(self, ctx: discord.ApplicationContext, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.respond(":x: You do not have the required permissions to use this command.", ephemeral=True)
        else:
            raise error

    @channels_setting.command(description="Remove a channel from the bot to respond")
    @commands.has_permissions(manage_guild=True)
    async def remove(self, ctx: discord.ApplicationContext, channel: Union[discord.TextChannel]):
        try:
            result = db.bot_settings.update_one(
                {"server_id": ctx.guild_id},
                {"$pull": {"channels": channel.id}}
            )
            if result.modified_count == 0:
                await ctx.respond(":warning: This channel was not found in the list.", ephemeral=True)
            else:
                await ctx.respond(f"<a:verificadoTESTE:799380003426795561> <#{channel.id}> has been **removed!**", ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while removing the channel: {e}", ephemeral=True)
            print(f"Error in channels_setting remove: {e}")

    @remove.error
    async def remove_error(self, ctx: discord.ApplicationContext, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.respond(":x: You do not have the required permissions to use this command.", ephemeral=True)
        else:
            raise error

    @channels_setting.command(description="Show a list of allowed channels in this server")
    async def list(self, ctx):
        try:
            server_settings = db.bot_settings.find_one({"server_id": ctx.guild_id})
            channels = server_settings.get("channels", []) if server_settings else []
            if channels:
                channel_mentions = ", ".join(f"<#{channel_id}>" for channel_id in channels)
                description = f"{channel_mentions}"
            else:
                description = "No channels have been added yet."
                
            list_embed = discord.Embed(
                title="List of allowed channels",
                description=description,
                color=discord.Colour.gold()
            )

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            list_embed.set_footer(text=f"Requested by {ctx.author.name} on {current_time}")
            await ctx.respond(embed=list_embed)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while listing channels: {e}", ephemeral=True)
            print(f"Error in listing channels: {e}")

    # change model command
    @change_model.command(description="Change the AI model used by the bot.")
    @option("model", description="Choose the model to use.", choices=["gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20", "gemini-2.5-flash-preview-04-17", "gemini-2.0-flash"])
    async def set(self, ctx: discord.ApplicationContext, model: str):
        try:
            user_settings = db.bot_settings.find_one({"user_id": ctx.author.id})
            if model in ["gemini-2.5-pro-preview-05-06"] and not user_settings.get("is_donator", False):
                await ctx.respond("<:info:1220157026739552296> This model is only available for donators.", ephemeral=True)
                return
            # Update the model in the database for this user id
            db.bot_settings.update_one(
                {"user_id": ctx.author.id},
                {"$set": {"model": model}},
                upsert=True
            )
            await ctx.respond(f"<a:verificadoTESTE:799380003426795561> AI model set to `{model}`.", ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while setting the model: {e}", ephemeral=True)
            print(f"Error in change_model: {e}")
    
    # command to set author mention preferences
    @settings.command(name="mention", description="Set whether the bot should mention the author in responses.")
    @option("mention", description="Choose whether to mention the author in responses (aka ping).", choices=[True, False])
    async def mention(self, ctx: discord.ApplicationContext, mention: bool):
        try:
            # Update the mention preference in the database for this user id
            db.bot_settings.update_one(
                {"user_id": ctx.author.id},
                {"$set": {"mention_author": mention}},
                upsert=True
            )
            if mention:
                await ctx.respond("<a:verificadoTESTE:799380003426795561> The bot will now mention you in responses.", ephemeral=True)
            else:
                await ctx.respond("<a:verificadoTESTE:799380003426795561> The bot will no longer mention you in responses.", ephemeral=True)
        except Exception as e:
            await ctx.respond(f":x: An error occurred while setting the mention preference: {e}", ephemeral=True)
            print(f"Error in mention command: {e}")

def setup(bot):
    bot.add_cog(Settings(bot))

# this will have some commands to change bot settings:
# - change AI temperature - check
# - change AI model - done 
# - Either if the bot should mention author or not - done
# - Allowed channels ID for the bot to respond in - check
