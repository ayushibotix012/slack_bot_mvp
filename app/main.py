# app/main.py

from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI
from .slack_listener import start_socket_mode

load_dotenv()

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    print("⚡ Starting Slack Socket Mode listener...")
    asyncio.create_task(start_socket_mode())


@app.get("/")
def root():
    return {"message": "StakeholderBot API is running"}


