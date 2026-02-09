"""
A.R.I.A. FastAPI Server
=======================
WebSocket-based real-time medical transcription and AI analysis.
"""

import asyncio
import json
import logging
import io
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.transcriber import get_transcriber, TranscriptSegment
from agent_graph import process_transcript_streaming, process_transcript

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="A.R.I.A. API",
    description="Automatic Reporting and Intelligent Analysis for Healthcare",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)
    
    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            await connection.send_json(data)


manager = ConnectionManager()


# =============================================================================
# Audio Processing
# =============================================================================

class AudioBuffer:
    """Buffer for accumulating audio chunks."""
    
    def __init__(self):
        self.chunks: list[bytes] = []
        self.total_duration: float = 0.0
    
    def add_chunk(self, chunk: bytes, duration_ms: int = 2000):
        self.chunks.append(chunk)
        self.total_duration += duration_ms / 1000
    
    def get_audio(self) -> bytes:
        return b"".join(self.chunks)
    
    def clear(self):
        self.chunks = []
        self.total_duration = 0.0


async def process_audio_chunk(audio_bytes: bytes, websocket: WebSocket) -> Optional[str]:
    """
    Process an audio chunk through Whisper.
    
    Args:
        audio_bytes: Raw audio data
        websocket: WebSocket for sending updates
    
    Returns:
        Transcribed text or None
    """
    try:
        transcriber = get_transcriber()
        segments = list(transcriber.transcribe_audio_chunk(audio_bytes))
        
        if segments:
            text = " ".join([seg.text for seg in segments])
            
            # Send transcript event
            await manager.send_json(websocket, {
                "type": "transcript",
                "data": {
                    "text": text,
                    "segments": [
                        {
                            "text": seg.text,
                            "start": seg.start,
                            "end": seg.end,
                            "confidence": seg.confidence
                        }
                        for seg in segments
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            return text
        
        return None
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        await manager.send_json(websocket, {
            "type": "error",
            "data": f"Transcription error: {str(e)}"
        })
        return None


async def run_agents(transcript: str, websocket: WebSocket):
    """
    Run the LangGraph agent pipeline with streaming updates.
    """
    try:
        # Unload Whisper to free VRAM for LLM
        transcriber = get_transcriber()
        transcriber.unload_model()
        
        await manager.send_json(websocket, {
            "type": "thought",
            "data": "Starting AI analysis pipeline..."
        })
        
        # Stream agent thoughts
        async for event in process_transcript_streaming(transcript):
            await manager.send_json(websocket, event)
            await asyncio.sleep(0.1)  # Small delay for UI updates
        
        await manager.send_json(websocket, {
            "type": "complete",
            "data": "Analysis complete"
        })
        
    except Exception as e:
        logger.error(f"Agent pipeline error: {e}")
        await manager.send_json(websocket, {
            "type": "error",
            "data": f"Agent error: {str(e)}"
        })


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/listen")
async def websocket_listen(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time transcription.
    
    Protocol:
    - Client sends binary audio chunks (WebM/Opus or WAV)
    - Client sends JSON control messages: {"action": "start"|"stop"|"process"}
    - Server sends JSON events: {"type": "transcript"|"thought"|"soap"|"error", "data": ...}
    """
    await manager.connect(websocket)
    
    audio_buffer = AudioBuffer()
    full_transcript: list[str] = []
    
    try:
        while True:
            data = await websocket.receive()
            
            # Handle binary audio data
            if "bytes" in data:
                audio_bytes = data["bytes"]
                audio_buffer.add_chunk(audio_bytes)
                
                # Process accumulated audio if we have enough
                if audio_buffer.total_duration >= 2.0:
                    text = await process_audio_chunk(audio_buffer.get_audio(), websocket)
                    if text:
                        full_transcript.append(text)
                    audio_buffer.clear()
            
            # Handle JSON control messages
            elif "text" in data:
                try:
                    message = json.loads(data["text"])
                    action = message.get("action")
                    
                    if action == "start":
                        audio_buffer.clear()
                        full_transcript.clear()
                        await manager.send_json(websocket, {
                            "type": "status",
                            "data": "Recording started"
                        })
                    
                    elif action == "stop":
                        # Process any remaining audio
                        if audio_buffer.chunks:
                            text = await process_audio_chunk(audio_buffer.get_audio(), websocket)
                            if text:
                                full_transcript.append(text)
                            audio_buffer.clear()
                        
                        await manager.send_json(websocket, {
                            "type": "status",
                            "data": "Recording stopped"
                        })
                    
                    elif action == "process":
                        # Run agent pipeline on accumulated transcript
                        if full_transcript:
                            combined = " ".join(full_transcript)
                            await run_agents(combined, websocket)
                        else:
                            await manager.send_json(websocket, {
                                "type": "error",
                                "data": "No transcript available to process"
                            })
                    
                    elif action == "test":
                        # Test with sample text
                        sample_text = message.get("text", "Patient has high sugars and BP issues")
                        await run_agents(sample_text, websocket)
                    
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# =============================================================================
# REST Endpoints
# =============================================================================

class TranscriptRequest(BaseModel):
    text: str


@app.post("/api/process")
async def process_text(request: TranscriptRequest):
    """
    Process text through the agent pipeline (non-streaming).
    """
    try:
        result = process_transcript(request.text)
        return JSONResponse({
            "success": True,
            "soap_note": result.get("soap_note", {}),
            "icd_codes": result.get("icd_codes", []),
            "fhir_compliant": result.get("fhir_compliant", False),
            "missing_fields": result.get("missing_info_flags", [])
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "A.R.I.A.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/models/status")
async def model_status():
    """Check loaded model status."""
    transcriber = get_transcriber()
    return {
        "whisper_loaded": transcriber._is_loaded,
        "whisper_model": transcriber.model_size,
        "compute_type": transcriber.compute_type
    }


# =============================================================================
# Startup/Shutdown
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("A.R.I.A. API starting up...")
    logger.info("WebSocket endpoint: ws://localhost:8000/ws/listen")


@app.on_event("shutdown")
async def shutdown():
    logger.info("A.R.I.A. API shutting down...")
    # Cleanup models
    transcriber = get_transcriber()
    transcriber.unload_model()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
