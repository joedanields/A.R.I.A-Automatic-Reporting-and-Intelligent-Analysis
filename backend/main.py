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
from services.procedure_suggester import get_procedure_suggester
from services.record_store import get_record_store
from services.patient_context import load_patient_context, get_patient_history_list
from services.learning_store import get_learning_store
from services.patient_summary import generate_patient_summary, get_supported_languages
from services.auth import get_auth_service
from services.audit import get_audit_log
from services.interaction_checker import get_interaction_checker
from services.diarization import get_diarization_service
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

    F5: Includes detected language per segment.
    F6: Includes speaker diarization when available.

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

            # F5: Include language info
            detected_lang = segments[0].language if segments else "en"
            lang_prob = segments[0].language_probability if segments else 1.0

            # F6: Run diarization if available
            speaker_segments = []
            try:
                diarization = get_diarization_service()
                if diarization.is_available():
                    speaker_segments = diarization.diarize(audio_bytes)
            except Exception as de:
                logger.debug(f"Diarization skipped: {de}")

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
                            "confidence": seg.confidence,
                            "language": seg.language,
                            "language_probability": seg.language_probability,
                        }
                        for seg in segments
                    ],
                    "detected_language": detected_lang,
                    "language_probability": lang_prob,
                    "speakers": [
                        {
                            "start": s.start,
                            "end": s.end,
                            "speaker": s.speaker,
                        }
                        for s in speaker_segments
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
    patient_id: str | None = None
    abha_id: str | None = None


@app.post("/api/process")
async def process_text(request: TranscriptRequest):
    """
    Process text through the agent pipeline (non-streaming).
    """
    try:
        result = process_transcript(
            request.text,
            patient_id=request.patient_id,
            abha_id=request.abha_id,
        )
        return JSONResponse({
            "success": True,
            "soap_note": result.get("soap_note", {}),
            "icd_codes": result.get("icd_codes", []),
            "procedure_codes": result.get("procedure_codes", []),
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
        **result,
    })


# =============================================================================
# Procedure/Billing Code Suggestion (F12)
# =============================================================================

@app.get("/api/procedures")
async def get_procedures():
    """Get procedure code database info (categories and counts)."""
    suggester = get_procedure_suggester()
    return JSONResponse({
        "success": True,
        "total_codes": len(suggester.get_all_codes()),
        "categories": suggester.list_categories(),
        "category_counts": {
            cat: len(suggester.search_by_category(cat))
            for cat in suggester.list_categories()
        },
    })


@app.get("/api/procedures/search")
async def search_procedures(q: str):
    """Search procedure codes by keyword."""
    suggester = get_procedure_suggester()
    results = []
    query_lower = q.lower()
    for proc in suggester.get_all_codes():
        keywords = [kw.lower() for kw in proc.get("keywords", [])]
        desc = proc.get("description", "").lower()
        if any(kw in query_lower or query_lower in kw for kw in keywords) or query_lower in desc:
            results.append(proc)
    return JSONResponse({
        "success": True,
        "query": q,
        "results": results,
    })


@app.get("/api/procedures/{category}")
async def get_procedures_by_category(category: str):
    """Get procedure codes in a specific category."""
    suggester = get_procedure_suggester()
    codes = suggester.search_by_category(category)
    return JSONResponse({
        "success": True,
        "category": category,
        "codes": codes,
    })


@app.post("/api/procedures/suggest")
async def suggest_procedures(request: TranscriptRequest):
    """Suggest procedure codes for a given transcript."""
    suggester = get_procedure_suggester()
    results = suggester.suggest(
        entities=[],
        transcript=request.text,
        n_results=5,
    )
    return JSONResponse({
        "success": True,
        "suggestions": results,
    })


# =============================================================================
# Encrypted Record Store (F15)
# =============================================================================

class RecordSaveRequest(BaseModel):
    transcript: str
    soap_note: dict
    icd_codes: list[dict] = []
    procedure_codes: list[dict] = []
    patient_id: str | None = None
    abha_id: str | None = None
    fhir_compliant: bool = False


@app.post("/api/records")
async def save_record(request: RecordSaveRequest):
    """Save a consultation record (encrypted at rest)."""
    try:
        store = get_record_store()
        result = store.save(
            transcript=request.transcript,
            soap_note=request.soap_note,
            icd_codes=request.icd_codes,
            procedure_codes=request.procedure_codes,
            patient_id=request.patient_id,
            abha_id=request.abha_id,
            fhir_compliant=request.fhir_compliant,
        )
        return JSONResponse({"success": True, "record": result})
    except Exception as e:
        logger.error(f"Record save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/records")
async def list_records(
    patient_id: str | None = None,
    abha_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List consultation records (metadata only)."""
    try:
        store = get_record_store()
        records = store.list_records(
            patient_id=patient_id,
            abha_id=abha_id,
            limit=limit,
            offset=offset,
        )
        total = store.count(patient_id=patient_id)
        return JSONResponse({
            "success": True,
            "total": total,
            "records": records,
        })
    except Exception as e:
        logger.error(f"Record list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/records/{record_id}")
async def get_record(record_id: str):
    """Retrieve and decrypt a record by ID."""
    try:
        store = get_record_store()
        record = store.get(record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        return JSONResponse({"success": True, "record": record})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Record retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/records/{record_id}")
async def delete_record(record_id: str):
    """Delete a record by ID."""
    try:
        store = get_record_store()
        deleted = store.delete(record_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Record not found")
        return JSONResponse({"success": True, "message": f"Record {record_id[:8]}... deleted"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Record deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/records/{record_id}/export")
async def export_record(record_id: str):
    """Export a record as FHIR-like JSON."""
    try:
        store = get_record_store()
        exported = store.export_record(record_id)
        if not exported:
            raise HTTPException(status_code=404, detail="Record not found")
        return JSONResponse({"success": True, "export": exported})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Record export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Patient Context / History (F16)
# =============================================================================

@app.get("/api/patients/{patient_id}/history")
async def get_patient_history(patient_id: str, limit: int = 20, offset: int = 0):
    """Get visit history for a patient."""
    try:
        records = get_patient_history_list(patient_id=patient_id, limit=limit, offset=offset)
        store = get_record_store()
        total = store.count(patient_id=patient_id)
        return JSONResponse({
            "success": True,
            "patient_id": patient_id,
            "total": total,
            "visits": records,
        })
    except Exception as e:
        logger.error(f"Patient history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients/{patient_id}/context")
async def get_patient_context(patient_id: str):
    """Get longitudinal patient context for the current consult."""
    try:
        context = load_patient_context(patient_id=patient_id)
        return JSONResponse({
            "success": True,
            "patient_id": patient_id,
            "has_history": bool(context),
            "context": context,
        })
    except Exception as e:
        logger.error(f"Patient context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Correction-as-Learning Loop (F4)
# =============================================================================

class CorrectionRequest(BaseModel):
    correction_type: str  # transcript, code, entity, drug
    original: str
    corrected: str
    context: str = ""
    entity_type: str = ""


@app.post("/api/learn/corrections")
async def add_learned_correction(request: CorrectionRequest):
    """Record a doctor correction for future learning."""
    try:
        store = get_learning_store()
        result = store.add_correction(
            correction_type=request.correction_type,
            original=request.original,
            corrected=request.corrected,
            context=request.context,
            entity_type=request.entity_type,
        )
        return JSONResponse({"success": True, "correction": result})
    except Exception as e:
        logger.error(f"Learning correction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learn/corrections")
async def list_learned_corrections(
    correction_type: str | None = None,
    limit: int = 100,
):
    """List learned corrections."""
    try:
        store = get_learning_store()
        corrections = store.get_corrections(
            correction_type=correction_type,
            limit=limit,
        )
        return JSONResponse({
            "success": True,
            "total": store.count(correction_type=correction_type),
            "corrections": corrections,
        })
    except Exception as e:
        logger.error(f"Learning list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learn/apply")
async def apply_learned_corrections(request: TranscriptRequest):
    """Apply learned corrections to text (preview)."""
    try:
        store = get_learning_store()
        corrected, applied = store.apply_corrections(request.text)
        return JSONResponse({
            "success": True,
            "original": request.text,
            "corrected": corrected,
            "applied": applied,
        })
    except Exception as e:
        logger.error(f"Learning apply error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/learn/corrections")
async def clear_learned_corrections(correction_type: str | None = None):
    """Clear learned corrections."""
    try:
        store = get_learning_store()
        count = store.clear_corrections(correction_type=correction_type)
        return JSONResponse({
            "success": True,
            "deleted": count,
        })
    except Exception as e:
        logger.error(f"Learning clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Patient-Facing Summary (F20)
# =============================================================================

class SummaryRequest(BaseModel):
    soap_note: dict
    language: str = "en"


@app.get("/api/summary/languages")
async def get_summary_languages():
    """Get supported languages for patient summaries."""
    return JSONResponse({
        "success": True,
        "languages": get_supported_languages(),
    })


@app.post("/api/summary/generate")
async def generate_summary(request: SummaryRequest):
    """Generate a plain-language patient summary from a SOAP note."""
    try:
        summary = generate_patient_summary(
            soap_note=request.soap_note,
            language=request.language,
        )
        return JSONResponse({
            "success": True,
            "language": request.language,
            "summary": summary,
        })
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Authentication (F18)
# =============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "doctor"
    display_name: str = ""


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Authenticate a user and return a token."""
    try:
        auth = get_auth_service()
        result = auth.authenticate(request.username, request.password)
        if not result:
            audit = get_audit_log()
            audit.log(
                action="login",
                username=request.username,
                outcome="failure",
                details="Invalid credentials",
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")

        audit = get_audit_log()
        audit.log(
            action="login",
            user_id=result["user"]["id"],
            username=result["user"]["username"],
            outcome="success",
        )
        return JSONResponse({"success": True, **result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/me")
async def get_current_user(authorization: str | None = None):
    """Get current user from token (header or query param)."""
    try:
        token = None
        if authorization:
            token = authorization.replace("Bearer ", "")

        if not token:
            raise HTTPException(status_code=401, detail="No token provided")

        auth = get_auth_service()
        user = auth.verify_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return JSONResponse({"success": True, "user": user})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/users")
async def create_user(request: CreateUserRequest):
    """Create a new user (admin only)."""
    try:
        auth = get_auth_service()
        user = auth.create_user(
            username=request.username,
            password=request.password,
            role=request.role,
            display_name=request.display_name,
        )
        audit = get_audit_log()
        audit.log(
            action="create_user",
            resource_type="user",
            resource_id=user["id"],
            details=f"Created user {request.username} (role={request.role})",
        )
        return JSONResponse({"success": True, "user": user})
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/users")
async def list_users():
    """List all active users."""
    try:
        auth = get_auth_service()
        users = auth.list_users()
        return JSONResponse({"success": True, "users": users})
    except Exception as e:
        logger.error(f"User list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auth/users/{user_id}")
async def deactivate_user(user_id: str):
    """Deactivate a user (admin only)."""
    try:
        auth = get_auth_service()
        deleted = auth.deactivate_user(user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="User not found")
        audit = get_audit_log()
        audit.log(
            action="deactivate_user",
            resource_type="user",
            resource_id=user_id,
        )
        return JSONResponse({"success": True, "message": f"User {user_id[:8]}... deactivated"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deactivation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Audit Log (F18)
# =============================================================================

@app.get("/api/audit")
async def query_audit_log(
    user_id: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    limit: int = 100,
):
    """Query audit log entries."""
    try:
        audit = get_audit_log()
        entries = audit.query(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            limit=limit,
        )
        return JSONResponse({
            "success": True,
            "total": audit.count(),
            "entries": entries,
        })
    except Exception as e:
        logger.error(f"Audit query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit/verify")
async def verify_audit_chain():
    """Verify the integrity of the audit chain."""
    try:
        audit = get_audit_log()
        is_valid, entries_checked = audit.verify_chain()
        return JSONResponse({
            "success": True,
            "chain_valid": is_valid,
            "entries_checked": entries_checked,
            "total_entries": audit.count(),
        })
    except Exception as e:
        logger.error(f"Audit verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vocab")
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
# Drug Interaction Checker (F10)
# =============================================================================

class InteractionCheckRequest(BaseModel):
    drug_names: list[str]


@app.get("/api/interactions")
async def get_interactions_info():
    """Get interaction database stats."""
    checker = get_interaction_checker()
    return JSONResponse({
        "success": True,
        **checker.get_stats(),
    })


@app.get("/api/interactions/search")
async def search_interaction_drugs(q: str):
    """Search drugs in interaction database (fuzzy)."""
    checker = get_interaction_checker()
    results = checker.search_drugs(q)
    return JSONResponse({
        "success": True,
        "query": q,
        "results": results,
    })


@app.get("/api/interactions/drug/{drug_name}")
async def get_drug_interactions(drug_name: str):
    """Get all interactions for a specific drug."""
    checker = get_interaction_checker()
    info = checker.get_drug_info(drug_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Drug '{drug_name}' not found")
    return JSONResponse({"success": True, **info})


@app.post("/api/interactions/check")
async def check_interactions(request: InteractionCheckRequest):
    """Check a list of drugs for interactions."""
    try:
        checker = get_interaction_checker()
        result = checker.check(request.drug_names)
        return JSONResponse({"success": True, **result})
    except Exception as e:
        logger.error(f"Interaction check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Speaker Diarization (F6)
# =============================================================================

@app.get("/api/diarization")
async def diarization_info():
    """Get diarization service status and model info."""
    service = get_diarization_service()
    return JSONResponse({
        "success": True,
        **service.get_model_info(),
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
