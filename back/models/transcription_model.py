from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

load_dotenv()


openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai.api_key)

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

MIME_EXT_MAP = {
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/amr": ".amr",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-ms-wma": ".wma",
}

def ext_from_mime(mime: str) -> str:
    if not mime:
        return ".bin"
    mime = mime.split(";")[0].strip().lower()
    return MIME_EXT_MAP.get(mime, ".bin")

def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    return transcript.text

def ext_from_mime(mime: str) -> str:
    if not mime:
        return ".bin"
    mime = mime.split(";")[0].strip().lower()
    return MIME_EXT_MAP.get(mime, ".bin")