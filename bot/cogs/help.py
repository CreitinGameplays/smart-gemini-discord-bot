import time
import discord
from discord.ext import commands
import datetime

class Help(commands.Cog):
    def __init__(self, bot: commands.Bot): 
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
        embed.set_footer(text=f"Requested by {ctx.author}", icon_url=ctx.user.avatar.url)
        embed.set_thumbnail(url=self.bot.user.avatar.url)

        await ctx.edit(embed=embed)

    @discord.slash_command(name="help", description="List all available bot commands or get help on a specific command")
    async def help(self, ctx, command: str = None):
        if command:
            target_command = None
            for cmd in self.bot.application_commands:
                if cmd.name.lower() == command.lower():
                    target_command = cmd
                    break
            if target_command:
                embed = discord.Embed(title=f"Help - /{target_command.name}", color=discord.Color.blurple())
                embed.description = target_command.description or "No description available."
                # If the command has options, list them
                if hasattr(target_command, "options") and target_command.options:
                    options_text = "\n".join(
                        f"**{opt.name}**: {opt.description}" for opt in target_command.options
                    )
                    embed.add_field(name="Options", value=options_text, inline=False)
                current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                embed.set_footer(text=f"Requested by {ctx.author} | {current_time}", icon_url=ctx.user.avatar.url)
                embed.set_thumbnail(url=self.bot.user.avatar.url)
                await ctx.respond(embed=embed)
            else:
                await ctx.respond(f":x: Command `{command}` not found.", ephemeral=True)
        else:
            # List all commands except help if no specific command was provided.
            commands_list = [
                f"/{cmd.name} - {cmd.description}"
                for cmd in self.bot.application_commands
                if cmd.name != "help"
            ]
            if not commands_list:
                commands_list.append("No commands available.")

            embed = discord.Embed(title="Help - List of Commands ⚙️", color=discord.Color.blurple())
            embed.description = "\n".join(commands_list)
            current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            embed.set_footer(text=f"Requested by {ctx.author} | {current_time}", icon_url=ctx.user.avatar.url)
            embed.set_thumbnail(url=self.bot.user.avatar.url)

            await ctx.respond(embed=embed)

def setup(bot):
    bot.add_cog(Help(bot))
