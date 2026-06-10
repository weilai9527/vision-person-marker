import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
except ImportError:
    RTCPeerConnection = None
    RTCSessionDescription = None

from services.camera_engine import process_camera_frame_image

logger = logging.getLogger(__name__)

_camera_webrtc_loop = None
_camera_webrtc_loop_lock = threading.Lock()
_camera_webrtc_pcs: dict[str, object] = {}
_camera_webrtc_results: dict[str, dict] = {}
_camera_webrtc_executor = ThreadPoolExecutor(max_workers=1)


def get_camera_webrtc_loop():
    global _camera_webrtc_loop
    with _camera_webrtc_loop_lock:
        if _camera_webrtc_loop is None or _camera_webrtc_loop.is_closed():
            _camera_webrtc_loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_camera_webrtc_loop.run_forever,
                name="camera-webrtc-loop",
                daemon=True,
            )
            thread.start()
        return _camera_webrtc_loop


async def consume_camera_webrtc_track(track, session_id: str, target: str, reset_tracking: bool, model_name: str) -> None:
    reset_next_frame = reset_tracking
    sequence = 0
    loop = asyncio.get_running_loop()
    while True:
        try:
            frame = await track.recv()
        except Exception:
            break

        sequence += 1
        image = frame.to_image().convert("RGB")
        try:
            result = await loop.run_in_executor(
                _camera_webrtc_executor,
                process_camera_frame_image,
                image,
                target,
                session_id,
                reset_next_frame,
                model_name,
            )
            reset_next_frame = False
            result["sequence"] = sequence
            result["transport"] = "webrtc"
            _camera_webrtc_results[session_id] = result
        except Exception as exc:
            logger.exception("WebRTC camera detection failed")
            _camera_webrtc_results[session_id] = {
                "success": False,
                "error": str(exc),
                "sequence": sequence,
                "transport": "webrtc",
            }


async def create_camera_webrtc_answer(
    offer_data: dict,
    session_id: str,
    target: str,
    reset_tracking: bool,
    model_name: str,
) -> dict:
    pc = RTCPeerConnection()

    old_pc = _camera_webrtc_pcs.pop(session_id, None)
    if old_pc is not None:
        await old_pc.close()
    _camera_webrtc_pcs[session_id] = pc
    _camera_webrtc_results.pop(session_id, None)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            asyncio.ensure_future(
                consume_camera_webrtc_track(track, session_id, target, reset_tracking, model_name),
                loop=asyncio.get_running_loop(),
            )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await pc.close()
            if _camera_webrtc_pcs.get(session_id) is pc:
                _camera_webrtc_pcs.pop(session_id, None)

    offer = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    for _ in range(50):
        if pc.iceGatheringState == "complete":
            break
        await asyncio.sleep(0.1)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def close_camera_webrtc_session(session_id: str) -> None:
    pc = _camera_webrtc_pcs.pop(session_id, None)
    _camera_webrtc_results.pop(session_id, None)
    if pc is not None:
        await pc.close()


def get_webrtc_result(session_id: str) -> dict | None:
    return _camera_webrtc_results.get(session_id)
