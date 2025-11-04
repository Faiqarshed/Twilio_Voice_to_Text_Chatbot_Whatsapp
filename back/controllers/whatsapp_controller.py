from models.transcription_model import transcribe_audio
from twilio.rest import Client
import requests
import os
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
import time
from openai import OpenAI
import openai
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from models.transcription_model import ext_from_mime
from controllers.rag_controller import query_rag_response

DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
TWILIO_ACCOUNT_SID = os.getenv("account_sid")
TWILIO_AUTH_TOKEN = os.getenv("auth_token")
ENABLE_TWILIO_VALIDATION = os.getenv("ENABLE_TWILIO_VALIDATION", "false").lower() in ("1","true","yes")


validator = RequestValidator(TWILIO_AUTH_TOKEN) if ENABLE_TWILIO_VALIDATION and TWILIO_AUTH_TOKEN else None


async def handle_whatsapp_webhook(request: Request):
    form = await request.form()
    form_dict = dict(form) #dict basically makes the form easier to acccess like 'form' has multiple fields (to, from, body, etc) it divides them into accessble .get fields like we can use .get to access their fields.

    num_media = int(form_dict.get("NumMedia", "0") or 0)
    resp = MessagingResponse()

    if num_media > 0:
        media_url = form_dict.get("MediaUrl0")
        media_mime = form_dict.get("MediaContentType0") or ""
        print(f"Received media: url={media_url} mime={media_mime} NumMedia={num_media}")

        ts = int(time.time() * 1000)
        msg_sid = form_dict.get("MessageSid", "msg")
        ext = ext_from_mime(media_mime)
        filename = f"{msg_sid}_{ts}{ext}"
        filepath = DOWNLOAD_DIR / filename

        #For downloading the media file from the media url
        r = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), stream=True)
        r.raise_for_status()

        with open(filepath, "wb") as f:
           for chunk in r.iter_content(chunk_size=8192):
               if chunk:
                  f.write(chunk)

        text = transcribe_audio(filepath)
        response = await query_rag_response(text)
        resp.message(f"Transcription: {response}")
        return Response(content=str(resp), media_type="application/xml")
    else:
        body = form_dict.get("Body", "")
        print(f"Text message from {form_dict.get('From')}: {body}")
        resp.message("Thanks — send a voice note and I'll capture it.")
        return Response(content=str(resp), media_type="application/xml")