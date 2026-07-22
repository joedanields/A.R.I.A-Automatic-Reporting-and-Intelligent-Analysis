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
from services.vocab_corrector import get_corrector
from services.drug_corrector import get_drug_corrector
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
# Evaluation Endpoint (F13)
# =============================================================================

class EvalRequest(BaseModel):
    case_ids: Optional[list[str]] = None


@app.post("/api/eval")
async def run_evaluation(request: EvalRequest = EvalRequest()):
    """
    Run evaluation harness on gold test cases.

    Returns metrics: WER, entity F1, code accuracy, SOAP similarity.
    Optionally filter to specific case_ids.
    """
    try:
        from services.eval_harness import EvalHarness

        harness = EvalHarness()
        result = harness.run_eval(request.case_ids)

        return JSONResponse({
            "success": True,
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "total_cases": result.total_cases,
            "passed_cases": result.passed_cases,
            "metrics": {
                "avg_wer": result.avg_wer,
                "avg_entity_f1": result.avg_entity_f1,
                "avg_code_accuracy": result.avg_code_accuracy,
                "avg_soap_similarity": result.avg_soap_similarity,
            },
            "duration_seconds": result.duration_seconds,
            "cases": [
                {
                    "case_id": cr.case_id,
                    "description": cr.description,
                    "passed": cr.passed,
                    "wer": cr.wer,
                    "entity_f1": cr.entity_f1,
                    "code_accuracy": cr.code_accuracy,
                    "soap_similarity": cr.soap_similarity,
                    "error": cr.error,
                }
                for cr in result.case_results
            ]
        })
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eval/history")
async def eval_history():
    """Get historical evaluation results."""
    try:
        from services.eval_harness import EvalHarness, RESULTS_DIR
        import os

        harness = EvalHarness()
        results = []

        if RESULTS_DIR.exists():
            for result_file in sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True):
                try:
                    with open(result_file, encoding="utf-8") as f:
                        data = json.load(f)
                    results.append({
                        "run_id": data.get("run_id"),
                        "timestamp": data.get("timestamp"),
                        "total_cases": data.get("total_cases"),
                        "passed_cases": data.get("passed_cases"),
                        "avg_wer": data.get("avg_wer"),
                        "avg_entity_f1": data.get("avg_entity_f1"),
                        "avg_code_accuracy": data.get("avg_code_accuracy"),
                        "duration_seconds": data.get("duration_seconds"),
                    })
                except Exception as e:
                    logger.error(f"Failed to load {result_file}: {e}")

        return JSONResponse({
            "success": True,
            "history": results[:50]  # Last 50 runs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Vocabulary Management (F7)
# =============================================================================

@app.get("/api/vocab")
async def get_vocab():
    """Get current clinic vocabulary (hotwords and corrections)."""
    try:
        corrector = get_corrector()
        return JSONResponse({
            "success": True,
            "categories": corrector.list_categories(),
            "hotwords": {
                cat: corrector.get_hotwords(cat)
                for cat in corrector.list_categories()
            },
            "corrections": corrector._corrections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vocab/hotwords")
async def get_hotwords(category: str | None = None):
    """Get hotwords, optionally filtered by category."""
    corrector = get_corrector()
    return JSONResponse({
        "success": True,
        "hotwords": corrector.get_hotwords(category),
    })


@app.post("/api/vocab/correct")
async def correct_text(request: TranscriptRequest):
    """Apply vocabulary corrections to text."""
    corrector = get_corrector()
    result = corrector.correct(request.text)
    return JSONResponse({
        "success": True,
        **result,
    })


class CorrectionRequest(BaseModel):
    misspelling: str
    correct: str


@app.post("/api/vocab/corrections")
async def add_correction(request: CorrectionRequest):
    """Add a new correction mapping."""
    corrector = get_corrector()
    corrector.add_correction(request.misspelling, request.correct)
    return JSONResponse({
        "success": True,
        "message": f"Added correction: {request.misspelling} -> {request.correct}",
    })


# =============================================================================
# Drug Name Correction (F8)
# =============================================================================

@app.get("/api/drugs")
async def get_drugs():
    """Get drug database info (categories and counts)."""
    corrector = get_drug_corrector()
    return JSONResponse({
        "success": True,
        "total_drugs": len(corrector._drug_names),
        "categories": corrector.list_categories(),
        "category_counts": {
            cat: len(drugs)
            for cat, drugs in corrector._drug_categories.items()
        },
    })


@app.get("/api/drugs/search")
async def search_drugs(q: str):
    """Search drugs by name (fuzzy)."""
    corrector = get_drug_corrector()
    results = corrector.search_drugs(q)
    return JSONResponse({
        "success": True,
        "query": q,
        "results": results,
    })


@app.get("/api/drugs/{category}")
async def get_drugs_by_category(category: str):
    """Get drugs in a specific category."""
    corrector = get_drug_corrector()
    drugs = corrector.get_drugs_in_category(category)
    return JSONResponse({
        "success": True,
        "category": category,
        "drugs": drugs,
    })


@app.post("/api/drugs/correct")
async def correct_drug_names(request: TranscriptRequest):
    """Apply drug-name corrections to text."""
    corrector = get_drug_corrector()
    result = corrector.correct(request.text)
    return JSONResponse({
        "success": True,
        **result,
    })


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
