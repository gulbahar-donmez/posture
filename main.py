from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
load_dotenv()

from database import engine, Base
from routers import auth, posture_analyzer, result, live_posture_ai_enhanced, report_ai, contact
from fastapi.middleware.cors import CORSMiddleware
from routers.chatbot import router as chatbot
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis

app = FastAPI(title="PostureGuard API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    redis_url = os.getenv("REDIS_URL")
    try:
        redis_connection = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(redis_connection)
        print("Redis connection successful and FastAPILimiter initialized.")
    except Exception as e:
        print(f"Could not connect to Redis: {e}")

# React build dosyaları varsa static olarak mount et
if os.path.exists("frontend/build/static"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

origins = [
    "http://35.193.229.80/",
    "https://35.193.229.80",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(posture_analyzer.router)
app.include_router(live_posture_ai_enhanced.router)
app.include_router(result.router)
app.include_router(report_ai.router)
app.include_router(chatbot)
app.include_router(contact.router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "PostureGuard API is running!", "version": "1.0.0"}

# React uygulaması için diğer tüm route'larda index.html döndür
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    if full_path.startswith(("api/", "docs", "openapi.json")):
        return {"error": "Not found"}

    index_path = "frontend/build/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "React app not found"}
