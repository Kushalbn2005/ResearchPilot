from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    database_url: str
    qdrant_url: str = "http://localhost:6333"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"  # verify it is still offered on your Groq account
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # 384-dim
    upload_dir: str = "uploads"
    max_upload_mb: int = 25
    max_pdf_pages: int = 60


settings = Settings()