"""python-bot: Core mínimo de bot con IA + memoria longitudinal.

Template base para cualquier plataforma (Discord, Telegram, CLI, API, etc).
Inspirado en SerenityOps (config) + free-intelligence/aurity.io (memoria).

Ejemplo de uso como CLI interactivo. Para integrar con Discord/Telegram/etc,
importa config, MemoryStore, y LLMClient desde tu propio entry point.
"""

import asyncio

from config import settings
from core import MemoryStore, LLMClient

CHANNEL = "cli"
USER_ID = "local"
USER_NAME = "user"


async def main():
    memory = MemoryStore(settings.db_path)
    await memory.connect()

    llm = LLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    print("python-bot ready | escribe 'salir' para terminar")
    print(f"modelo: {settings.llm_model} | memoria: {settings.db_path}\n")

    try:
        while True:
            message = input("tú: ").strip()
            if not message or message.lower() in ("salir", "exit", "quit"):
                break

            if message == "/stats":
                stats = await memory.get_stats(CHANNEL)
                print(f"  mensajes: {stats['total_messages']} | usuarios: {stats['unique_users']}\n")
                continue

            if message.startswith("/buscar "):
                query = message[8:]
                results = await memory.search(CHANNEL, query, limit=10)
                if not results:
                    print("  sin resultados\n")
                else:
                    for r in results:
                        print(f"  {r['user_name']}: {r['content'][:100]}")
                    print()
                continue

            # Guardar mensaje
            await memory.store(CHANNEL, USER_ID, USER_NAME, "user", message)

            # Contexto jerárquico: reciente + relevante
            recent = await memory.get_recent(CHANNEL, settings.memory_recent_limit)
            relevant = await memory.search(CHANNEL, message, settings.memory_relevant_limit)
            context = memory.build_context(recent, relevant)

            # LLM
            response = await llm.chat(settings.system_prompt, context)

            # Guardar respuesta
            await memory.store(CHANNEL, "bot", "assistant", "assistant", response)

            print(f"bot: {response}\n")

    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
