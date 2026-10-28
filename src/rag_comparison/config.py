from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    openrouter_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    model: str = Field(default="openai/gpt-4.1-mini", alias="RAGBENCH_MODEL")
    embedding_model: str = Field(
        default="openai/text-embedding-3-small", alias="RAGBENCH_EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=1536, gt=0, alias="RAGBENCH_EMBEDDING_DIM")
    neo4j_uri: str = Field(default="bolt://127.0.0.1:17687", alias="RAGBENCH_NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="RAGBENCH_NEO4J_USER")
    neo4j_password: SecretStr = Field(
        default=SecretStr("ragbench-local-only"), alias="RAGBENCH_NEO4J_PASSWORD"
    )
    data_dir: Path = Field(default=Path(".ragbench"), alias="RAGBENCH_DATA_DIR")
    runs_dir: Path = Field(default=Path("runs"), alias="RAGBENCH_RUNS_DIR")
    concurrency: int = Field(default=4, ge=1, le=32, alias="RAGBENCH_CONCURRENCY")
    request_timeout: float = Field(default=120, gt=0, alias="RAGBENCH_REQUEST_TIMEOUT")

    def public(self) -> dict:
        return {
            "model": self.model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "openrouter_base_url": self.openrouter_base_url,
            "concurrency": self.concurrency,
        }

    def require_key(self) -> None:
        if not self.openrouter_api_key.get_secret_value():
            raise ValueError("Set OPENROUTER_API_KEY in your shell or a local .env file.")
