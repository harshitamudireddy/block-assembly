import os
import secrets
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
CV_API_KEY = os.getenv("CV_API_KEY", "assembly-local-key").strip()
DROIDCAM_DEFAULT_URL = os.getenv("DROIDCAM_URL", "http://192.168.2.217:4747/video").strip()

USE_LOCAL_MODE = not (bool(SUPABASE_URL) and bool(SUPABASE_SERVICE_ROLE_KEY))

if USE_LOCAL_MODE:
    print("\n=======================================================")
    print(" [MODE] Running in LOCAL / STANDALONE in-memory mode   ")
    print(" Access Web Dashboard: http://localhost:8000           ")
    print("=======================================================\n")
else:
    print(f"[MODE] Connected to Supabase at {SUPABASE_URL}")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Assembly Performance API",
    description="Backend API and Dashboard for Live Block Assembly Inspection",
    version="1.3.0",
)

# Allow React / Hatchable dashboard to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOCAL IN-MEMORY STORAGE & FRAME BUFFER
# ============================================================

LOCAL_OPERATORS = {
    "OP001": {"id": "op_001", "operator_code": "OP001", "name": "Harshal (Station Lead)", "role": "lead"},
    "OP002": {"id": "op_002", "operator_code": "OP002", "name": "Operator 2", "role": "operator"},
}

LOCAL_SESSIONS = {}  # cycle_id -> dict
LOCAL_EVENTS = []    # list of event dicts

LATEST_HUD_FRAME: Optional[bytes] = None
LATEST_FRAME_TIME: float = 0.0
LATEST_ACTIVE_CYCLE_ID: Optional[str] = None
DAILY_GOAL: int = 50
ACTIVE_OPERATOR: Optional[dict] = None
RESET_REQUESTED: bool = False


# ============================================================
# SUPABASE HELPERS
# ============================================================

SUPABASE_REST_URL = f"{SUPABASE_URL.rstrip('/')}/rest/v1" if SUPABASE_URL else ""


SUPABASE_SESSION = requests.Session()
_supabase_adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10)
SUPABASE_SESSION.mount("https://", _supabase_adapter)
SUPABASE_SESSION.mount("http://", _supabase_adapter)


def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def supabase_request(
    method: str,
    table: str,
    params: Optional[dict] = None,
    json_data=None,
):
    """
    Communicate with Supabase REST API.
    """
    if USE_LOCAL_MODE:
        return None

    url = f"{SUPABASE_REST_URL}/{table}"

    response = SUPABASE_SESSION.request(
        method=method,
        url=url,
        headers={
            **supabase_headers(),
            "Prefer": "return=representation",
        },
        params=params,
        json=json_data,
        timeout=10,
    )

    if not response.ok:
        print(f"SUPABASE ERROR [{method} {table}]: {response.status_code} {response.text}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Supabase request failed",
                "status": response.status_code,
                "response": response.text,
            },
        )

    if not response.text:
        return None

    return response.json()


def resolve_operator(identifier: Optional[str] = None) -> Optional[dict]:
    global ACTIVE_OPERATOR
    if USE_LOCAL_MODE:
        target = identifier or (ACTIVE_OPERATOR.get("operator_code") if ACTIVE_OPERATOR else "OP001")
        for op in LOCAL_OPERATORS.values():
            if op["id"] == target or op["operator_code"].upper() == target.upper() or op["name"].lower() == target.lower():
                return op
        new_op = {
            "id": f"op_{target.lower().replace(' ', '_')}",
            "operator_code": target.upper()[:10],
            "name": f"Operator {target}",
            "role": "operator",
        }
        LOCAL_OPERATORS[target] = new_op
        return new_op

    target = identifier or (ACTIVE_OPERATOR.get("id") if ACTIVE_OPERATOR else None) or "OP001"

    # Check if target looks like a UUID
    is_uuid = len(target) == 36 and target.count("-") == 4

    # 1. Try matching by UUID if valid UUID format
    if is_uuid:
        try:
            ops = supabase_request(
                "GET",
                "operators",
                params={
                    "id": f"eq.{target}",
                    "select": "id,operator_code,name,role",
                    "limit": "1",
                },
            )
            if ops and len(ops) > 0:
                return ops[0]
        except Exception:
            pass

    # 2. Try matching by operator_code
    try:
        ops = supabase_request(
            "GET",
            "operators",
            params={
                "operator_code": f"eq.{target}",
                "select": "id,operator_code,name,role",
                "limit": "1",
            },
        )
        if ops and len(ops) > 0:
            return ops[0]
    except Exception:
        pass

    # 3. Try matching by name
    try:
        ops = supabase_request(
            "GET",
            "operators",
            params={
                "name": f"ilike.{target}",
                "select": "id,operator_code,name,role",
                "limit": "1",
            },
        )
        if ops and len(ops) > 0:
            return ops[0]
    except Exception:
        pass

    if ACTIVE_OPERATOR:
        return ACTIVE_OPERATOR

    # Fallback to first operator in database
    try:
        all_ops = supabase_request(
            "GET",
            "operators",
            params={"select": "id,operator_code,name,role", "limit": "1"},
        )
        if all_ops and len(all_ops) > 0:
            return all_ops[0]
    except Exception:
        pass

    return None


# ============================================================
# AUTHENTICATION
# ============================================================

def verify_cv_api_key(api_key: Optional[str]):
    """
    Verify incoming requests from the CV inspection system.
    If CV_API_KEY is default 'assembly-local-key' or empty, permit local connection.
    Otherwise enforce strict secret comparison.
    """
    if not CV_API_KEY or CV_API_KEY == "assembly-local-key":
        return

    if not api_key or not secrets.compare_digest(api_key, CV_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid CV API key",
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class SetOperatorRequest(BaseModel):
    operator_id: str


class StartAssemblyRequest(BaseModel):
    operator_id: Optional[str] = Field(
        None,
        description="Operator code or UUID, e.g. OP001 or OP002",
    )


class EventRequest(BaseModel):
    cycle_id: str

    state_index: Optional[int] = None
    state_name: Optional[str] = None
    state_title: Optional[str] = None

    status: str = Field(
        ...,
        description="holding, advanced, error or completed",
    )

    confidence: Optional[float] = None
    is_valid: Optional[bool] = None

    diagnostic: Optional[str] = None
    consensus: Optional[str] = None
    error_type: Optional[str] = None

    incoming_object: Optional[str] = None
    incoming_confidence: Optional[float] = None
    incoming_expected: Optional[bool] = None

    detections: Optional[dict] = None
    spatial_checks: Optional[dict] = None
    part_counts: Optional[dict] = None


class EndAssemblyRequest(BaseModel):
    cycle_id: str

    status: str = Field(
        ...,
        description="pass or fail",
    )

    failure_reason: Optional[str] = None

    states_completed: Optional[int] = None
    total_states: int = 9


class GoalRequest(BaseModel):
    goal: int = Field(50, ge=1, le=1000)


class ResetAssemblyRequest(BaseModel):
    operator_id: Optional[str] = "OP001"


# ============================================================
# HEALTH & CAMERA STREAM PROXIES
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "assembly-performance-api",
        "mode": "local" if USE_LOCAL_MODE else "supabase",
        "supabase_configured": not USE_LOCAL_MODE,
        "hud_stream_active": bool(LATEST_HUD_FRAME and (time.time() - LATEST_FRAME_TIME < 3.0)),
    }


@app.get("/api/assembly/operator")
def get_active_operator():
    global ACTIVE_OPERATOR
    if not ACTIVE_OPERATOR:
        ACTIVE_OPERATOR = resolve_operator(None)
    return {"status": "ok", "operator": ACTIVE_OPERATOR}


@app.post("/api/assembly/operator")
def set_active_operator(data: SetOperatorRequest):
    global ACTIVE_OPERATOR
    op = resolve_operator(data.operator_id)
    if not op:
        raise HTTPException(status_code=404, detail=f"Operator '{data.operator_id}' not found")
    ACTIVE_OPERATOR = op

    # If any assembly session is currently in progress, update its operator_id
    if USE_LOCAL_MODE:
        for sess in LOCAL_SESSIONS.values():
            if sess.get("status") == "in_progress":
                sess["operator_id"] = op["id"]
                sess["operator_code"] = op.get("operator_code", "")
                sess["operator_name"] = op.get("name", "")
    else:
        try:
            active_sessions = supabase_request(
                "GET",
                "assembly_sessions",
                params={"status": "eq.in_progress", "select": "id", "limit": "10"},
            )
            if active_sessions:
                for s in active_sessions:
                    supabase_request(
                        "PATCH",
                        "assembly_sessions",
                        params={"id": f"eq.{s['id']}"},
                        json_data={"operator_id": op["id"]},
                    )
        except Exception as e:
            print(f"[Operator Sync Warning]: {e}")

    return {"status": "ok", "operator": ACTIVE_OPERATOR}


@app.post("/api/camera/frame")
async def receive_frame(
    request: Request,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    """
    Receives compressed JPEG HUD frames from live_demo.py with all YOLO
    bounding boxes, labels, and PiP insets.
    """
    global LATEST_HUD_FRAME, LATEST_FRAME_TIME, RESET_REQUESTED
    verify_cv_api_key(x_cv_api_key)
    LATEST_HUD_FRAME = await request.body()
    LATEST_FRAME_TIME = time.time()

    reset_needed = RESET_REQUESTED
    RESET_REQUESTED = False

    return {
        "status": "ok",
        "reset_requested": reset_needed,
        "operator_code": ACTIVE_OPERATOR.get("operator_code") if ACTIVE_OPERATOR else None,
        "operator_id": ACTIVE_OPERATOR.get("id") if ACTIVE_OPERATOR else None,
    }


@app.get("/api/camera/hud_stream")
def hud_stream():
    """
    Streams the live AI-annotated HUD frame (with bounding boxes, PiP, checklist)
    to browser clients as a standard MJPEG stream.
    """
    def generate():
        last_sent_time = 0.0
        while True:
            try:
                if LATEST_HUD_FRAME and (time.time() - LATEST_FRAME_TIME < 3.0):
                    # Yield when a new frame arrives or periodic heartbeat
                    if LATEST_FRAME_TIME != last_sent_time or (time.time() - last_sent_time > 1.0):
                        last_sent_time = LATEST_FRAME_TIME
                        frame_data = LATEST_HUD_FRAME
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                            + frame_data
                            + b"\r\n"
                        )
                time.sleep(0.02)
            except (GeneratorExit, StopIteration):
                break
            except Exception:
                break

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/camera/stream")
def camera_stream():
    """
    Proxies the DroidCam raw MJPEG stream directly to browser clients.
    """
    def generate():
        try:
            req = urllib.request.urlopen(DROIDCAM_DEFAULT_URL, timeout=3.0)
            while True:
                chunk = req.read(4096)
                if not chunk:
                    break
                yield chunk
        except Exception:
            return

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=--dcmjpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ============================================================
# START ASSEMBLY
# ============================================================

@app.post("/api/assembly/start")
def start_assembly(
    data: StartAssemblyRequest,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)
    op = resolve_operator(data.operator_id)
    if not op:
        op = {"id": "op_001", "operator_code": "OP001", "name": "Station Lead", "role": "operator"}

    # --------------------------------------------------------
    # Local Mode Execution
    # --------------------------------------------------------
    if USE_LOCAL_MODE:
        # Close any previous in_progress sessions
        for s in LOCAL_SESSIONS.values():
            if s.get("status") == "in_progress":
                s["status"] = "fail"
                s["failure_reason"] = "Superceded by new cycle"
                s["end_time"] = datetime.now(timezone.utc).isoformat()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(2).upper()
        cycle_id = f"ASM-{timestamp}-{random_part}"
        session_id = f"sess_{secrets.token_hex(4)}"
        cycle_number = len(LOCAL_SESSIONS) + 1

        session_data = {
            "id": session_id,
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
            "operator_id": op["id"],
            "operator_code": op.get("operator_code", "OP001"),
            "operator_name": op.get("name", "Operator"),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
            "failure_reason": None,
        }
        LOCAL_SESSIONS[cycle_id] = session_data

        return {
            "success": True,
            "message": "Assembly started (Local Mode)",
            "assembly": {
                "id": session_id,
                "cycle_id": cycle_id,
                "cycle_number": cycle_number,
                "operator_id": op.get("operator_code", "OP001"),
                "operator_name": op.get("name", "Operator"),
                "status": "in_progress",
            },
        }

    # --------------------------------------------------------
    # Supabase Mode Execution
    # --------------------------------------------------------
    # Close any existing in_progress sessions first
    try:
        active_sessions = supabase_request(
            "GET",
            "assembly_sessions",
            params={"status": "eq.in_progress", "select": "id", "limit": "10"},
        )
        if active_sessions:
            for s in active_sessions:
                supabase_request(
                    "PATCH",
                    "assembly_sessions",
                    params={"id": f"eq.{s['id']}"},
                    json_data={
                        "status": "fail",
                        "failure_reason": "Superceded by new cycle",
                        "end_time": datetime.now(timezone.utc).isoformat(),
                    },
                )
    except Exception as e:
        print(f"[Supabase start cleanup warning]: {e}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(2).upper()
    cycle_id = f"ASM-{timestamp}-{random_part}"

    assembly = supabase_request(
        "POST",
        "assembly_sessions",
        params={
            "select": (
                "id,"
                "cycle_id,"
                "operator_id,"
                "start_time,"
                "status,"
                "states_completed,"
                "total_states"
            )
        },
        json_data={
            "cycle_id": cycle_id,
            "operator_id": op["id"],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
        },
    )

    if not assembly:
        raise HTTPException(
            status_code=500,
            detail="Failed to create assembly session",
        )

    return {
        "success": True,
        "message": "Assembly started",
        "assembly": {
            "id": assembly[0]["id"],
            "cycle_id": cycle_id,
            "operator_id": op.get("operator_code") or op["id"],
            "operator_name": op.get("name"),
            "status": "in_progress",
        },
    }


# ============================================================
# ASSEMBLY EVENT
# ============================================================

@app.post("/api/assembly/event")
def assembly_event(
    data: EventRequest,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)

    # --------------------------------------------------------
    # Local Mode Execution
    # --------------------------------------------------------
    if USE_LOCAL_MODE:
        global LATEST_ACTIVE_CYCLE_ID
        LATEST_ACTIVE_CYCLE_ID = data.cycle_id

        session = LOCAL_SESSIONS.get(data.cycle_id)
        if not session:
            cycle_number = len(LOCAL_SESSIONS) + 1
            session = {
                "id": f"sess_{secrets.token_hex(4)}",
                "cycle_id": data.cycle_id,
                "cycle_number": cycle_number,
                "operator_id": "op_001",
                "operator_code": "OP001",
                "operator_name": "Harshal (Station Lead)",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,
                "duration_seconds": 0,
                "status": "in_progress",
                "states_completed": 0,
                "total_states": 9,
                "failure_reason": None,
            }
            LOCAL_SESSIONS[data.cycle_id] = session

        event_id = f"evt_{secrets.token_hex(4)}"
        event_record = {
            "id": event_id,
            "assembly_id": session["id"],
            "cycle_id": data.cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state_index": data.state_index,
            "state_name": data.state_name,
            "state_title": data.state_title,
            "status": data.status,
            "confidence": data.confidence,
            "is_valid": data.is_valid,
            "diagnostic": data.diagnostic,
            "consensus": data.consensus,
            "error_type": data.error_type,
            "incoming_object": data.incoming_object,
            "incoming_confidence": data.incoming_confidence,
            "incoming_expected": data.incoming_expected,
            "detections": data.detections,
            "spatial_checks": data.spatial_checks,
            "part_counts": data.part_counts,
        }
        LOCAL_EVENTS.append(event_record)

        if data.state_index is not None:
            session["states_completed"] = data.state_index
            session["current_state_index"] = data.state_index
        else:
            session["states_completed"] = session.get("states_completed", 0)

        return {
            "success": True,
            "cycle_id": data.cycle_id,
            "event_id": event_id,
            "states_completed": session["states_completed"],
        }

    # --------------------------------------------------------
    # Supabase Mode Execution
    # --------------------------------------------------------
    assemblies = supabase_request(
        "GET",
        "assembly_sessions",
        params={
            "cycle_id": f"eq.{data.cycle_id}",
            "select": (
                "id,"
                "cycle_id,"
                "states_completed,"
                "total_states,"
                "status"
            ),
            "limit": "1",
        },
    )

    if not assemblies:
        raise HTTPException(
            status_code=404,
            detail=f"Assembly '{data.cycle_id}' not found",
        )

    assembly = assemblies[0]

    event_data = {
        "assembly_id": assembly["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state_index": data.state_index,
        "state_name": data.state_name,
        "state_title": data.state_title,
        "status": data.status,
        "confidence": data.confidence,
        "is_valid": data.is_valid,
        "diagnostic": data.diagnostic,
        "consensus": data.consensus,
        "error_type": data.error_type,
        "incoming_object": data.incoming_object,
        "incoming_confidence": data.incoming_confidence,
        "incoming_expected": data.incoming_expected,
        "detections": data.detections,
        "spatial_checks": data.spatial_checks,
        "part_counts": data.part_counts,
    }

    event = supabase_request(
        "POST",
        "assembly_events",
        params={"select": "id,assembly_id,timestamp,status,state_index"},
        json_data=event_data,
    )

    current_completed = assembly.get("states_completed") or 0
    new_completed = current_completed

    if (
        data.status in ["advanced", "completed", "holding"]
        and data.state_index is not None
    ):
        if data.state_index == 0:
            new_completed = 0
        else:
            new_completed = max(
                current_completed,
                data.state_index,
            )

    if new_completed != current_completed:
        supabase_request(
            "PATCH",
            "assembly_sessions",
            params={
                "id": f"eq.{assembly['id']}",
            },
            json_data={
                "states_completed": new_completed,
            },
        )

    return {
        "success": True,
        "cycle_id": data.cycle_id,
        "event_id": event[0]["id"] if event else None,
        "states_completed": new_completed,
    }


# ============================================================
# END ASSEMBLY
# ============================================================

@app.post("/api/assembly/end")
def end_assembly(
    data: EndAssemblyRequest,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    verify_cv_api_key(x_cv_api_key)

    # --------------------------------------------------------
    # Local Mode Execution
    # --------------------------------------------------------
    if USE_LOCAL_MODE:
        session = LOCAL_SESSIONS.get(data.cycle_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Assembly '{data.cycle_id}' not found",
            )

        start_time = datetime.fromisoformat(session["start_time"].replace("Z", "+00:00"))
        end_time = datetime.now(timezone.utc)
        duration_seconds = max(0, int((end_time - start_time).total_seconds()))

        status = data.status.lower()
        if status not in ["pass", "fail"]:
            raise HTTPException(status_code=400, detail="status must be 'pass' or 'fail'")

        session["end_time"] = end_time.isoformat()
        session["duration_seconds"] = duration_seconds
        session["status"] = status
        session["states_completed"] = (
            data.states_completed if data.states_completed is not None else session.get("states_completed", 0)
        )
        session["total_states"] = data.total_states
        session["failure_reason"] = data.failure_reason

        return {
            "success": True,
            "assembly": session,
        }

    # --------------------------------------------------------
    # Supabase Mode Execution
    # --------------------------------------------------------
    assemblies = supabase_request(
        "GET",
        "assembly_sessions",
        params={
            "cycle_id": f"eq.{data.cycle_id}",
            "select": (
                "id,"
                "cycle_id,"
                "start_time,"
                "states_completed,"
                "total_states"
            ),
            "limit": "1",
        },
    )

    if not assemblies:
        raise HTTPException(
            status_code=404,
            detail=f"Assembly '{data.cycle_id}' not found",
        )

    assembly = assemblies[0]

    start_time = datetime.fromisoformat(
        assembly["start_time"].replace("Z", "+00:00")
    )
    end_time = datetime.now(timezone.utc)

    duration_seconds = max(
        0,
        int((end_time - start_time).total_seconds()),
    )

    states_completed = (
        data.states_completed
        if data.states_completed is not None
        else assembly.get("states_completed", 0)
    )

    status = data.status.lower()
    if status not in ["pass", "fail"]:
        raise HTTPException(
            status_code=400,
            detail="status must be 'pass' or 'fail'",
        )

    supabase_request(
        "PATCH",
        "assembly_sessions",
        params={
            "id": f"eq.{assembly['id']}",
        },
        json_data={
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "status": status,
            "states_completed": states_completed,
            "total_states": data.total_states,
            "failure_reason": data.failure_reason,
        },
    )

    return {
        "success": True,
        "assembly": {
            "id": assembly["id"],
            "cycle_id": data.cycle_id,
            "status": status,
            "duration_seconds": duration_seconds,
            "states_completed": states_completed,
            "total_states": data.total_states,
            "failure_reason": data.failure_reason,
            "end_time": end_time.isoformat(),
        },
    }


# ============================================================
# GOAL & RESET ENDPOINTS
# ============================================================

@app.post("/api/assembly/goal")
def set_goal(data: GoalRequest):
    global DAILY_GOAL
    DAILY_GOAL = data.goal
    return {"success": True, "goal_target": DAILY_GOAL}


@app.post("/api/assembly/reset")
def reset_assembly(
    data: Optional[ResetAssemblyRequest] = None,
    x_cv_api_key: Optional[str] = Header(default=None),
):
    global RESET_REQUESTED
    verify_cv_api_key(x_cv_api_key)
    # Only request CV client to reset if the command originated from dashboard frontend
    if not x_cv_api_key:
        RESET_REQUESTED = True

    op_code = (data.operator_id if data else None)
    op = resolve_operator(op_code) or {
        "id": "op_001",
        "operator_code": "OP001",
        "name": "Station Lead",
        "role": "operator",
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(2).upper()
    cycle_id = f"ASM-{timestamp}-{random_part}"

    if USE_LOCAL_MODE:
        # Close any previous in_progress sessions
        for s in LOCAL_SESSIONS.values():
            if s.get("status") == "in_progress":
                s["status"] = "fail"
                s["failure_reason"] = "Reset to Step 0"
                s["end_time"] = datetime.now(timezone.utc).isoformat()

        session_id = f"sess_{secrets.token_hex(4)}"
        cycle_number = len(LOCAL_SESSIONS) + 1

        session_data = {
            "id": session_id,
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
            "operator_id": op["id"],
            "operator_code": op.get("operator_code", "OP001"),
            "operator_name": op.get("name", "Operator"),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
            "failure_reason": None,
        }
        LOCAL_SESSIONS[cycle_id] = session_data

        event_id = f"evt_{secrets.token_hex(4)}"
        event_record = {
            "id": event_id,
            "assembly_id": session_id,
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state_index": 0,
            "state_name": "state_0_unstarted",
            "state_title": "0. Unstarted",
            "status": "holding",
            "confidence": 1.0,
            "is_valid": True,
            "diagnostic": "Station Ready for Next Unit",
            "consensus": "state_0_unstarted",
            "error_type": None,
            "incoming_object": None,
            "incoming_confidence": None,
            "incoming_expected": None,
            "detections": None,
            "spatial_checks": None,
            "part_counts": None,
        }
        LOCAL_EVENTS.append(event_record)

        return {
            "success": True,
            "message": "Assembly reset to Step 0 (Local Mode)",
            "assembly": {
                "id": session_id,
                "cycle_id": cycle_id,
                "cycle_number": cycle_number,
                "operator_id": op.get("operator_code", "OP001"),
                "operator_name": op.get("name", "Operator"),
                "status": "in_progress",
                "states_completed": 0,
            },
        }

    # Supabase Mode Execution
    # 1. Close any existing in_progress session as reset
    try:
        active_sessions = supabase_request(
            "GET",
            "assembly_sessions",
            params={"status": "eq.in_progress", "select": "id", "limit": "10"},
        )
        if active_sessions:
            for s in active_sessions:
                supabase_request(
                    "PATCH",
                    "assembly_sessions",
                    params={"id": f"eq.{s['id']}"},
                    json_data={
                        "status": "fail",
                        "failure_reason": "Reset to Step 0",
                        "end_time": datetime.now(timezone.utc).isoformat(),
                    },
                )
    except Exception as e:
        print(f"[Supabase reset cleanup warning]: {e}")

    # 2. Insert new session at Step 0
    now_iso = datetime.now(timezone.utc).isoformat()
    new_session = supabase_request(
        "POST",
        "assembly_sessions",
        params={
            "select": (
                "id,"
                "cycle_id,"
                "operator_id,"
                "start_time,"
                "status,"
                "states_completed,"
                "total_states"
            )
        },
        json_data={
            "cycle_id": cycle_id,
            "operator_id": op["id"],
            "start_time": now_iso,
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
        },
    )

    if not new_session:
        raise HTTPException(status_code=500, detail="Failed to create assembly session on reset")

    # 3. Insert Step 0 event
    supabase_request(
        "POST",
        "assembly_events",
        params={"select": "id"},
        json_data={
            "assembly_id": new_session[0]["id"],
            "timestamp": now_iso,
            "state_index": 0,
            "state_name": "state_0_unstarted",
            "state_title": "0. Unstarted",
            "status": "holding",
            "confidence": 1.0,
            "is_valid": True,
            "diagnostic": "Station Ready for Next Unit",
        },
    )

    return {
        "success": True,
        "message": "Assembly reset to Step 0",
        "assembly": {
            "id": new_session[0]["id"],
            "cycle_id": cycle_id,
            "operator_id": op.get("operator_code") or op["id"],
            "operator_name": op.get("name"),
            "status": "in_progress",
            "states_completed": 0,
            "total_states": 9,
        },
    }


# ============================================================
# QUERY ENDPOINTS FOR DASHBOARD
# ============================================================

@app.get("/api/assembly/sessions")
def get_sessions():
    if USE_LOCAL_MODE:
        return list(LOCAL_SESSIONS.values())[::-1]
    return supabase_request("GET", "assembly_sessions", params={"order": "start_time.desc", "limit": "50"})


@app.get("/api/assembly/events")
def get_events(cycle_id: Optional[str] = None):
    if USE_LOCAL_MODE:
        if cycle_id:
            return [e for e in LOCAL_EVENTS if e.get("cycle_id") == cycle_id]
        return LOCAL_EVENTS[-50:][::-1]
    params = {"order": "timestamp.desc", "limit": "50"}
    if cycle_id:
        sess = supabase_request("GET", "assembly_sessions", params={"cycle_id": f"eq.{cycle_id}", "select": "id", "limit": "1"})
        if sess and len(sess) > 0:
            params["assembly_id"] = f"eq.{sess[0]['id']}"
        else:
            return []
    return supabase_request("GET", "assembly_events", params=params)


@app.get("/api/assembly/latest")
def get_latest():
    if USE_LOCAL_MODE:
        sessions = list(LOCAL_SESSIONS.values())
        latest_event = LOCAL_EVENTS[-1] if LOCAL_EVENTS else None

        active_cid = None
        if latest_event and latest_event.get("cycle_id") in LOCAL_SESSIONS:
            active_cid = latest_event["cycle_id"]
        elif LATEST_ACTIVE_CYCLE_ID and LATEST_ACTIVE_CYCLE_ID in LOCAL_SESSIONS:
            active_cid = LATEST_ACTIVE_CYCLE_ID

        if active_cid and active_cid in LOCAL_SESSIONS:
            latest_session = LOCAL_SESSIONS[active_cid]
        elif sessions:
            latest_session = sessions[-1]
        else:
            latest_session = None

        current_step = 0
        if latest_event and latest_event.get("state_index") is not None:
            current_step = latest_event["state_index"]
        elif latest_session and latest_session.get("states_completed") is not None:
            current_step = latest_session["states_completed"]

        passes = sum(1 for s in sessions if s.get("status") == "pass")
        fails = sum(1 for s in sessions if s.get("status") == "fail")
        total = len(sessions)
        pass_rate = round((passes / total) * 100, 1) if total > 0 else 100.0

        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sessions_today = [
            s for s in sessions
            if (s.get("start_time") or "").startswith(today_prefix)
        ]
        cycles_done_today = sum(
            1 for s in sessions_today
            if s.get("status") in ["pass", "fail", "completed"]
        )
        today_passes = sum(1 for s in sessions_today if s.get("status") == "pass")
        today_fails = sum(1 for s in sessions_today if s.get("status") == "fail")

        active_cycle_number = latest_session.get("cycle_number") if latest_session else (len(sessions) if sessions else 1)
        hud_active = bool(LATEST_HUD_FRAME and (time.time() - LATEST_FRAME_TIME < 3.0))

        return {
            "session": latest_session,
            "latest_event": latest_event,
            "current_step": current_step,
            "total_cycles": total,
            "active_cycle_number": active_cycle_number,
            "cycles_done_today": cycles_done_today,
            "goal_target": DAILY_GOAL,
            "today_passes": today_passes,
            "today_fails": today_fails,
            "pass_count": passes,
            "fail_count": fails,
            "pass_rate": pass_rate,
            "hud_active": hud_active,
        }
    return {"status": "ok", "mode": "supabase"}



# ============================================================
# ROOT REDIRECT TO FRONTEND DASHBOARD
# ============================================================

@app.get("/")
def root():
    """Redirects to the dedicated React/Vite dashboard running in frontend/."""
    return RedirectResponse(url="http://localhost:8080/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
