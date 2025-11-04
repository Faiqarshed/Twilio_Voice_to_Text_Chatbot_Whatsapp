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
from controllers.whatsapp_controller import handle_whatsapp_webhook
from controllers.rag_controller import add_documents
from fastapi.responses import JSONResponse
from controllers.rag_controller import query_rag_response


app = FastAPI()

client = OpenAI()

@app.post("/twilio/whatsapp")
async def twilio_whatsapp_webhook(request: Request):
    return await handle_whatsapp_webhook(request)


@app.post("/rag/adddoc")
async def add_document_endpoint(request: Request):
    data = await request.json()
    docs = data.get("documents", [])
    await add_documents(docs)
    return JSONResponse({"status": "success", "added": len(docs)})


@app.post("/rag/query")
async def query_rag_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "")
    response = await query_rag_response(query)
    return JSONResponse({"query": query, "response": response})


