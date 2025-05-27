import discord
from discord.ext import commands
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os

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
        "set_temperature",
        "Set the AI temperature value (0-2)."
    )
    
    @settings.command(name="info", description="Displays bot settings information.")
    async def info(self, ctx: discord.ApplicationContext):
        await ctx.respond("Use one of the subcommands to manage bot settings.", ephemeral=True)

    @set_temperature.command(name="set_temperature", description="Set the AI temperature value (0-2).")
    async def settemperature(self, ctx: discord.ApplicationContext, value: int):
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

def setup(bot):
    bot.add_cog(Settings(bot))

# this will have some commands to change bot settings:
# - change AI temperature
# - change AI model
# - Either if the bot should mention author or not
# - Allowed channels ID for the bot to respond in