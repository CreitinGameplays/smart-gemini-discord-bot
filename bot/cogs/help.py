import time
import discord
from discord.ext import commands
import datetime
import psutil
import platform
import asyncio
import traceback
import logging

class AddMe(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        button = discord.ui.Button(label="Support Discord Server!", style=discord.ButtonStyle.link, url="https://discord.com/invite/Hg578mck2e")
        button2 = discord.ui.Button(label="Add me!", style=discord.ButtonStyle.link, url="https://discord.com/oauth2/authorize?client_id=1219407466526146661&scope=bot&permissions=277025704960", emoji="❤")
        self.add_item(button)
        self.add_item(button2)

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
                await ctx.respond(embed=embed, view=AddMe())
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
            await ctx.respond(embed=embed, view=AddMe())

    @discord.slash_command(
        name='about',
        description='Show detailed and technical information about the bot',
        integration_types={discord.IntegrationType.guild_install}
    )
    async def about(self, ctx: discord.ApplicationContext):
        self.is_slash_command = True
        await self._show_about(ctx)

    async def _show_about(self, ctx: discord.ApplicationContext):
        try:
            # Get system info
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            cpu_model = platform.processor()
    
            # Calculate total members and guilds
            total_members = sum(guild.member_count for guild in self.bot.guilds)
            total_guilds = len(self.bot.guilds)
        
            # Get bot specific info
            total_shards = self.bot.shard_count if self.bot.shard_count else 1
            current_shard = ctx.guild.shard_id if ctx.guild else 0
            current_shard_name = f"Shard {current_shard}"  # Adjust as needed
            shard_latency = self.bot.get_shard(current_shard).latency if self.bot.shard_count > 0 else self.bot.latency
    
            # Create embed
            embed = discord.Embed(
                title="Bot Information",
                color=0x58D68D
            )
    
            # Add bot owner info
            app_info = await self.bot.application_info()
            owner = app_info.owner
            embed.set_author(
                name=f"Made with ❤️ by {owner.name}",
                icon_url=owner.display_avatar.url
            )
    
            embed.add_field(
                name="Py-cord version",
                value=f"`{discord.__version__}`",
                inline=True
            )
    
            # Shard Information
            embed.add_field(
                name="Sharding Info",
                value=(f"```\n"
                       f"Total Shards: {total_shards}\n"
                       f"Current Shard: {current_shard_name} (ID: {current_shard})\n"
                       f"```"),
                inline=False
            )
    
            # Latency Information
            embed.add_field(
                name="Latency",
                value=(f"```\n"
                       f"Bot Latency: {round(self.bot.latency * 1000)}ms\n"
                       f"Shard Latency: {round(shard_latency * 1000)}ms\n"
                       f"```"),
                inline=False
            )
    
            # System Information
            embed.add_field(
                name="System Info",
                value=(f"```\n"
                       f"CPU: {cpu_model}\n"
                       f"CPU Cores: {cpu_count}\n"
                       f"CPU Usage: {cpu_usage}%\n"
                       f"Total RAM: {memory.total / (1024 ** 3):.1f}GB\n"
                       f"RAM Usage: {memory.percent}%\n"
                       f"```"),
                inline=False
            )
    
            # Bot Stats
            embed.add_field(
                name="Bot Stats",
                value=(f"```\n"
                       f"Total Servers: {total_guilds}\n"
                       f"Total Members: {total_members}\n"
                       f"```"),
                inline=False
            )
            
            # Send response
            await ctx.defer()
            msg = await ctx.respond(embed=embed, view=AddMe())
            await asyncio.sleep(20)
            await msg.delete()
    
        except Exception as e:
            logging.error("About command error:\n" + traceback.format_exc())
            await ctx.respond(f":x: An error occurred: `{e}`")

def setup(bot):
    bot.add_cog(Help(bot))
