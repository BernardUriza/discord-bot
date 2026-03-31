"""Discord bot con memoria longitudinal + Claude API.

Creado desde template python-bot (BernardUriza/python-bot).
"""

import discord
from discord.ext import commands

from config import settings
from core import MemoryStore, LLMClient

# --- Init ---
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=settings.command_prefix, intents=intents)
memory = MemoryStore(settings.db_path)
llm = LLMClient(
    api_key=settings.anthropic_api_key,
    model=settings.llm_model,
    max_tokens=settings.llm_max_tokens,
)


# --- Events ---
@bot.event
async def on_ready():
    await memory.connect()
    print(f"{bot.user} conectado | {len(bot.guilds)} servers")


# --- Commands ---
@bot.command(name="chat")
async def chat(ctx: commands.Context, *, message: str):
    """Habla con la IA. Usa memoria longitudinal del canal."""
    channel_id = str(ctx.channel.id)
    user_name = ctx.author.display_name

    await memory.store(channel_id, str(ctx.author.id), user_name, "user", message)

    recent = await memory.get_recent(channel_id, settings.memory_recent_limit)
    relevant = await memory.search(channel_id, message, settings.memory_relevant_limit)
    context = memory.build_context(recent, relevant)

    async with ctx.typing():
        response = await llm.chat(settings.system_prompt, context)

    await memory.store(channel_id, str(bot.user.id), bot.user.name, "assistant", response)

    for chunk in [response[i : i + 1990] for i in range(0, len(response), 1990)]:
        await ctx.send(chunk)


@bot.command(name="memoria")
async def memoria(ctx: commands.Context):
    """Muestra stats de memoria del canal."""
    stats = await memory.get_stats(str(ctx.channel.id))
    await ctx.send(
        f"**Memoria de este canal:**\n"
        f"- Mensajes: {stats['total_messages']}\n"
        f"- Usuarios: {stats['unique_users']}"
    )


@bot.command(name="buscar")
async def buscar(ctx: commands.Context, *, query: str):
    """Busca en la memoria longitudinal del canal."""
    results = await memory.search(str(ctx.channel.id), query, limit=10)
    if not results:
        await ctx.send("No encontré nada.")
        return

    lines = [f"**{r['user_name']}:** {r['content'][:100]}" for r in results]
    await ctx.send(f"**Resultados para '{query}':**\n" + "\n".join(lines))


@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.send(f"Pong! {round(bot.latency * 1000)}ms")


# --- Cleanup ---
@bot.event
async def on_close():
    await memory.close()


bot.run(settings.discord_token)
