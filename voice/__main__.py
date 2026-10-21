"""Chrysalis's voice service: speech in and out for the web UI's voice mode, one WebSocket per session (talk.py). It
keeps no preferences: each session names the parts it wants (parts.py), and the page's Settings are where they are
chosen. The web server starts it when a voice session opens and stops it once the last one has been gone a while, so it
takes no memory the rest of the time. It loads what a session asks for when it asks: voice mode's hello the whole stack,
a voice's sample (a character's voice picker) only that voice's model. It exits when its stdin ends (the web server is
gone) or on SIGTERM.
  data/voice-venv/bin/python -m voice [--port N] [--threads N]     prints "voice: ws://127.0.0.1:PORT" once it listens
threads (CHRYSALIS_VOICE_THREADS, 4) caps the CPU threads a model runs on: at 16 the speech models took the CPU
Flash-Next's experts need (95 tok/s down to 3), at 4 they cost it 12%."""
import argparse, asyncio, json, os, signal, sys, threading
from concurrent.futures import ThreadPoolExecutor

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .parts import Library
from .talk import Talk


async def main(port, threads):
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, os._exit, 0)
    # read until EOF: the parent's end of the pipe closes when it exits, however it exits
    threading.Thread(target=lambda: (sys.stdin.buffer.read(), os._exit(0)), daemon=True).start()
    lib, listen_pool, speak_pool = Library(threads), ThreadPoolExecutor(1), ThreadPoolExecutor(1)

    async def session(ws):
        async def send(x):
            try:
                await ws.send(x if isinstance(x, bytes) else json.dumps(x))
            except ConnectionClosed:
                pass
        talk = Talk(lib, send, listen_pool, speak_pool)
        try:
            async for m in ws:
                if isinstance(m, bytes):
                    talk.hear(m[:len(m) & ~1])
                    continue
                try:
                    msg = json.loads(m)
                except ValueError:
                    msg = None
                await talk.message(msg if isinstance(msg, dict) else {"type": None})
        except ConnectionClosed:
            pass
        finally:
            talk.close()

    async with serve(session, "127.0.0.1", port, compression=None, max_size=1 << 22) as server:
        print(f"voice: ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m voice")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CHRYSALIS_VOICE_THREADS") or 4))
    a = ap.parse_args()
    asyncio.run(main(a.port, max(1, a.threads)))
