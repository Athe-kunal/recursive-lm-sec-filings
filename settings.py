from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # FastAPI server URL
    server_url: str = "http://127.0.0.1:8888"


env_settings = EnvSettings()

# Tool server FastAPI route paths (typically joined with env_settings.server_url).
SEC_FILING_TOOL_ENDPOINT = "/tools/sec_filings_to_embed_and_search"
EARNINGS_TRANSCRIPT_TOOL_ENDPOINT = "/tools/earnings_transcript_to_embed_and_search"
