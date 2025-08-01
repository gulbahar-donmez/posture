from fastapi import FastAPI
from database import engine, Base
from routers import auth, posture_analyzer,result,live_posture_ai_enhanced, report_ai, contact
from fastapi.middleware.cors import CORSMiddleware
from routers.chatbot import router as chatbot
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis

Base.metadata.create_all(bind=engine)
app = FastAPI(title="PostureGuard API", version="1.0.0")

@app.on_event("startup")
async def startup():
    redis_connection = redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3001",
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