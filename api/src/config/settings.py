from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8001
    jwt_secret: str

    users_internal_url: str
    friends_internal_url: str
    internal_secret: str

    embeddings_db_path: str = "/app/data/embeddings.db"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    hf_api_token: str

    users_db_host: str
    users_db_port: int = 5432
    users_db_name: str = "postgres"
    users_db_user: str
    users_db_password: str


settings = Settings()
