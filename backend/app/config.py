import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "IntelliMed Longitudinal Health Intelligence Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database
    # Default to sqlite locally if DATABASE_URL is not set, to ensure the app works immediately
    DATABASE_URL: str = "sqlite:///./health_intelligence.db"

    # Authentication
    JWT_SECRET_KEY: str = "supersecretkeychangeinproduction1234567890!"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Cloudinary File Storage
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # Google Gemini API
    GEMINI_API_KEY: str = ""

    # Upstash Redis / Message Broker
    # Default to local sqlite celery broker or local redis if not set
    REDIS_URL: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
