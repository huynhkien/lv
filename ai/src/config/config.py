from pydantic_settings import BaseSettings, SettingsConfigDict

class Configs(BaseSettings):
    MONGO_URI: str
    CLIENT_URL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")  

settings = Configs()
