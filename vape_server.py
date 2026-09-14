"""
Live view of the vape rig: runs the rig in a background thread and streams
every window to the 3D page over a websocket.

    python vape_server.py                 # http://localhost:4670
    python vape_server.py --toxicity 2    # any vape_rig.Config field is a flag

GET  /                  the page (web/vape3d.html), in live mode
GET  /neurons.bin       float32 xyz per neuron
GET  /neurons_meta.bin  uint8 superclass code, uint8 placement code, per neuron
GET  /neurons.json      n, superclass names, highlight groups
GET  /state             the latest window's telemetry as JSON
GET  /replay/...        the last finished run's replay files, if any
WS   /stream            one binary message per window (format below)

Message: u32 header_len | header JSON (space-padded to 4 bytes) |
         u32 nF | u32[nF] fired | u32 nD | u32[nD] newly dead |
         u32 nR | u32[nR] at-risk index | u8[nR] damage 0..255 (zero-padded to 4 bytes)
The first message after connecting is a "sync": every dead neuron so far and
the recent telemetry, so a page opened mid-run is complete.
"""
import argparse
import asyncio
import collections
import json
import os
import struct
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
LIVE_DIR = ROOT / "build" / "vape_live"
PAGE = ROOT / "web" / "vape3d.html"


def pack(header, fired=None, dead=None, risk_idx=None, risk_lvl=None):
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 4)
    parts = [struct.pack("<I", len(h)), h]
    for arr in (fired, dead, risk_idx):
        a = np.zeros(0, dtype="<u4") if arr is None else np.asarray(arr, dtype="<u4")
        parts += [struct.pack("<I", len(a)), a.tobytes()]
    lvl = np.zeros(0, dtype=np.uint8) if risk_lvl is None else np.asarray(risk_lvl, dtype=np.uint8)
    parts.append(lvl.tobytes() + b"\0" * (-len(lvl) % 4))
    return b"".join(parts)


class Broadcast:
    def __init__(self, keep=600):
        self.lock = threading.Lock()
        self.frames = collections.deque(maxlen=keep)     # (seq, bytes)
        self.headers = collections.deque(maxlen=keep)
        self.seq = 0
        self.dead = []
        self.tests = []
        self.latest = {}
        self.summary = None
        self.status = "loading"

    def push(self, header, fired, new_dead, risk_idx, risk_lvl):
        header = dict(header, type="frame")
        msg = pack(header, fired, new_dead, risk_idx, risk_lvl)
        with self.lock:
            self.seq += 1
            self.frames.append((self.seq, msg))
            self.headers.append(header)
            if len(new_dead):
                self.dead.append(np.asarray(new_dead, dtype=np.uint32))
            if header.get("test"):
                self.tests.append(header["test"])
            self.latest = header

    def since(self, last):
        with self.lock:
            if not self.frames:
                return [], last                # nothing streamed yet
            if self.frames[0][0] > last + 1:
                return None, self.seq          # too far behind: resync
            return [m for s, m in self.frames if s > last], self.seq

    def sync(self):
        with self.lock:
            dead = np.concatenate(self.dead) if self.dead else np.zeros(0, np.uint32)
            header = {"type": "sync", "status": self.status, "history": list(self.headers)[-400:],
                      "tests": list(self.tests), "summary": self.summary}
            return pack(header, None, dead), self.seq


def make_app(fb, rig_cfg, record, meta_dir):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    import vape_rig

    bus = Broadcast()
    app = FastAPI()

    def run_rig():
        try:
            rig = vape_rig.Rig(fb, rig_cfg, on_frame=bus.push,
                               record_dir=vape_rig.RUN_DIR if record else None)
            app.state.rig = rig
            bus.status = "running"
            summary = rig.run(record=record)
            bus.summary = summary
            if record:
                vape_rig.build_replay(vape_rig.RUN_DIR, vape_rig.REPLAY_DIR)
            bus.status = "finished"
        except Exception:
            bus.status = "error: " + traceback.format_exc(limit=3)
            print(bus.status)
        with bus.lock:
            bus.seq += 1
            bus.frames.append((bus.seq, pack({"type": "status", "status": bus.status,
                                              "summary": bus.summary})))

    @app.on_event("startup")
    def start():
        threading.Thread(target=run_rig, daemon=True, name="rig").start()

    @app.get("/")
    def page():
        html = PAGE.read_text(encoding="utf-8").replace(
            'window.FLYVAPE_MODE = "replay"', 'window.FLYVAPE_MODE = "live"', 1)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    for name, kind in (("neurons.bin", "application/octet-stream"),
                       ("neurons_meta.bin", "application/octet-stream"),
                       ("neurons.json", "application/json")):
        def serve(name=name, kind=kind):
            return FileResponse(meta_dir / name, media_type=kind)
        app.add_api_route("/" + name, serve, methods=["GET"])

    @app.get("/state")
    def state():
        return JSONResponse({"status": bus.status, "latest": bus.latest, "tests": bus.tests,
                             "summary": bus.summary})

    vape_rig.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/replay", StaticFiles(directory=vape_rig.REPLAY_DIR), name="replay")
    models = ROOT / "build" / "models"          # reduced fly_model.bin / vape_model.bin, if any
    models.mkdir(parents=True, exist_ok=True)
    app.mount("/models", StaticFiles(directory=models), name="models")

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            msg, last = bus.sync()
            await ws.send_bytes(msg)
            while True:
                await asyncio.sleep(0.04)
                msgs, seq = bus.since(last)
                if msgs is None:
                    msg, last = bus.sync()
                    await ws.send_bytes(msg)
                    continue
                for m in msgs:
                    await ws.send_bytes(m)
                last = seq
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def main(argv=None):
    import vape
    import vape_rig
    import plume_fly

    ap = argparse.ArgumentParser(description="Live 3D view of the vape rig")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 4670)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-record", action="store_true")
    for f in vape_rig.Config.__dataclass_fields__.values():
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = vape_rig.Config(**{k: getattr(args, k) for k in vape_rig.Config.__dataclass_fields__})

    t = time.time()
    fb = vape_rig.load_brain()
    print(f"brain: {fb.n:,} neurons in {time.time() - t:.1f}s; writing anatomy ...")
    meta = vape.write_anatomy(fb, LIVE_DIR, plume_fly.motor_groups(fb, cfg.sim_steps))
    print(f"anatomy: {meta['placed_counts']}")

    import uvicorn
    app = make_app(fb, cfg, record=not args.no_record, meta_dir=LIVE_DIR)
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
