import time
import discord
from discord.ext import commands

class Help(commands.Cog):
    def __init__(self, bot): 
        self.bot = bot

    @discord.slash_command(name="ping", description="Shows the bot latency")
    @commands.cooldown(1, 1, commands.BucketType.user)
    async def ping(self, ctx):
        start_time = time.perf_counter()

        bot_latency_ms = round(self.bot.latency * 1000)
        total_shards = self.bot.shard_count if self.bot.shard_count else 1

        guild_shard = ctx.guild.shard_id if ctx.guild else 0

        shard_latency_ms = bot_latency_ms

        for shard_info in self.bot.latencies:
            shard_id, latency = shard_info
            if shard_id == guild_shard:
                shard_latency_ms = round(latency * 1000)
                break

        embed = discord.Embed(title="Pong! :ping_pong:", color=discord.Color.green())
        embed.add_field(name="Bot Latency", value=f"{bot_latency_ms} ms", inline=False)
        embed.add_field(name="Total Shards", value=str(total_shards), inline=False)
        embed.add_field(name="Current Shard", value=f"ID: {guild_shard} | Latency: {shard_latency_ms} ms", inline=False)

        await ctx.respond(embed=embed)
        end_time = time.perf_counter()
        message_latency_ms = round((end_time - start_time) * 1000)

        embed.add_field(name="Message Latency", value=f"{message_latency_ms} ms", inline=False)
        embed.set_footer(text=f"Requested by {ctx.author}", icon_url=self.bot.user.avatar.url)
        embed.set_thumbnail(url=self.bot.user.avatar.url)

        await ctx.edit_original_message(embed=embed)

def setup(bot):
    bot.add_cog(Help(bot))
