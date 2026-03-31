"""Configuración centralizada via .env + Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Discord
    discord_token: str = ""
    command_prefix: str = "!"

    # Anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024
    system_prompt: str = "You are a helpful assistant."

    # Memory
    memory_recent_limit: int = 20
    memory_relevant_limit: int = 5

    # Paths
    storage_dir: Path = Path(__file__).parent / "storage"
    db_path: Path = Path(__file__).parent / "storage" / "memory.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_dirs(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
