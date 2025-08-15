import sys
import os
import asyncio
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import redis

# --- NEW: Import SQLAlchemy Session for type hinting ---
from sqlalchemy.orm import Session

# Path fix to allow 'ml' imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports (Updated) ---
from backend.config import (
    REDIS_HOST,
    REDIS_PORT,
    INGEST_STREAM_NAME,
    RESULT_CHANNEL_NAME,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from backend.inference import InferenceWrapper
# get_user is now database-aware
from backend.security import User, get_user, verify_password, create_access_token
# get_db is our new dependency for database sessions
from backend.database import get_db

# --- Prometheus Metrics Imports ---
from prometheus_client import Counter, Gauge, make_asgi_app

# --- FastAPI App Instantiation ---
app = FastAPI(title="Predictive Maintenance API", version="1.1.0-db")

# --- Prometheus Metrics Setup (Unchanged) ---
PDM_HTTP_INGEST_REQUESTS_TOTAL = Counter(
    "pdm_http_ingest_requests_total",
    "Total number of HTTP ingest requests to /api/v1/stream"
)
PDM_WEBSOCKET_CONNECTIONS_ACTIVE = Gauge(
    "pdm_websocket_connections_active",
    "Number of active WebSocket connections"
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- Middleware (Unchanged) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Connection Manager (Unchanged) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        PDM_WEBSOCKET_CONNECTIONS_ACTIVE.inc()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            PDM_WEBSOCKET_CONNECTIONS_ACTIVE.dec()

    async def broadcast(self, message: str):
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                to_remove.append(connection)
        for conn in to_remove:
            self.disconnect(conn)

manager = ConnectionManager()

# --- Background Tasks & Startup Event (Unchanged) ---
async def redis_subscriber(manager: ConnectionManager, redis_client: redis.Redis):
    pubsub = redis_client.pubsub()
    pubsub.subscribe(RESULT_CHANNEL_NAME)
    while True:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if message and message.get("data"):
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            await manager.broadcast(data)
        await asyncio.sleep(0.01)

inference_wrapper: InferenceWrapper = None
redis_client = None

@app.on_event("startup")
async def startup_event():
    global inference_wrapper, redis_client
    inference_wrapper = InferenceWrapper()
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client = redis.Redis(connection_pool=pool)
    asyncio.create_task(redis_subscriber(manager, redis_client))

# --- API Router for v1 Endpoints ---
api_router = APIRouter(prefix="/api/v1")

# --- MODIFIED: Login Endpoint with Database Integration ---
@api_router.post("/auth/token", tags=["Authentication"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)  # <-- Injects the database session
):
    """
    Authenticates a user against the database and returns a JWT access token.
    """
    # get_user now queries the real database via the injected session
    user = get_user(db, username=form_data.username)

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Other Endpoints (Unchanged) ---
@api_router.get("/health", tags=["Monitoring"])
async def health_check():
    return {"status": "ok"}

class SensorData(BaseModel):
    machine_id: str
    sensors: Dict[str, float] = Field(..., example={
        "vibration_x": 0.0, "vibration_y": 0.0, "temperature": 0.0, "current": 0.0
    })

@api_router.post("/stream", tags=["Ingestion"])
async def stream_sensor_data(payload: SensorData):
    try:
        PDM_HTTP_INGEST_REQUESTS_TOTAL.inc()
        serialized_payload = payload.json()
        redis_client.xadd(INGEST_STREAM_NAME, {'data': serialized_payload})
        return {"status": "ok"}
    except Exception as e:
        error_msg = {"error": str(e)}
        return JSONResponse(status_code=500, content=error_msg)

# --- Final Setup ---
app.include_router(api_router)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")