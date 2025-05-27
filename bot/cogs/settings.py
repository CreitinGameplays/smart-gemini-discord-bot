import discord
from discord.ext import commands
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))


class Settings(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    """
    TODO: Add commands to change bot settings.
    """

def setup(bot):
    bot.add_cog(Settings(bot))

# this will have some commands to change bot settings:
# - change AI temperature
# - change AI model
# - Either if the bot should mention author or not
# - idk more