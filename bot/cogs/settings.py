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

async def setup_mongodb():
    try:
        # Check if the database exists
        db = mongo_client["gemini-bot-db"]
        c_list = mongo_client.list_collections()
        if 'bot_settings' not in c_list:
            print("Creating 'bot_settings' database...")
            await db.create_collection("bot_settings")
            return "Database 'gemini_bot' created successfully."
        else:
            print("'gemini_bot' database already exists.")
            return "Database 'gemini_bot' already exists."
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

def setup(bot):
    bot.add_cog(Settings(bot))

# this will have some commands to change bot settings:
# - change AI temperature
# - change AI model
# - Either if the bot should mention author or not
# - idk more