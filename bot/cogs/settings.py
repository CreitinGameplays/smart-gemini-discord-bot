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

    @settings.command(name="info", description="Manage bot settings.")
    async def open(self, ctx: discord.ApplicationContext):
        await ctx.respond("Use one of the subcommands to manage bot settings.", ephemeral=True)

    @set_temperature.command(description="Set the AI temperature value (0-2).")
    async def set(self, ctx: discord.ApplicationContext, value: float):
        if value < 0 or value > 2:
            await ctx.respond(":x: Temperature must be between 0 and 2.", ephemeral=True)
            return
        try:
            # Update the temperature in the database for this user id
            db.bot_settings.update_one(
                {"user_id": ctx.author.id},
                {"$set": {"temperature": value}},
                upsert=True
            )
            await ctx.respond(f"✅ AI temperature set to {value}.", ephemeral=True)
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
                await ctx.respond(f"✅ <#{channel.id}> has been **added!**", ephemeral=True)
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
                await ctx.respond(f"✅ <#{channel.id}> has been **removed!**", ephemeral=True)
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

def setup(bot):
    bot.add_cog(Settings(bot))

# this will have some commands to change bot settings:
# - change AI temperature - check
# - change AI model
# - Either if the bot should mention author or not
# - Allowed channels ID for the bot to respond in - check