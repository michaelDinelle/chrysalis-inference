"""kiln's voice service: speech in and out for the web UI's voice mode, one WebSocket per session (talk.py). It keeps no
preferences: each session names the parts it wants (parts.py), and the page's Settings are where they are chosen. The
web server starts it when a voice session opens and stops it once the last one has been gone a while, so it takes no
memory the rest of the time. It loads the default parts as it starts (the first session most likely wants them), and
exits when its stdin ends (the web server is gone) or on SIGTERM.
  data/voice-venv/bin/python -m voice [--port N] [--threads N]     prints "voice: ws://127.0.0.1:PORT" once it listens
threads (KILN_VOICE_THREADS, 4) caps the CPU threads a model runs on: at 16 the speech models took the CPU Flash-Next's
experts need (95 tok/s down to 3), at 4 they cost it 12%."""
import argparse, asyncio, json, os, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .parts import DEFAULTS, Library
from .talk import Talk


def rss_mb():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) // 1024


async def main(port, threads):
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, os._exit, 0)
    # read until EOF: the parent's end of the pipe closes when it exits, however it exits
    threading.Thread(target=lambda: (sys.stdin.buffer.read(), os._exit(0)), daemon=True).start()
    lib, listen_pool, speak_pool = Library(threads), ThreadPoolExecutor(1), ThreadPoolExecutor(1)

    def warm():
        t, held = time.monotonic(), []
        with ThreadPoolExecutor(len(DEFAULTS)) as ex:
            for kind, f in [(k, ex.submit(lib.get, k, n)) for k, n in DEFAULTS.items()]:
                try:
                    f.result()
                    held.append(kind)
                except Exception as e:
                    print(f"voice: {e}", file=sys.stderr, flush=True)
        print(f"voice: {', '.join(DEFAULTS[k] for k in held)} loaded in {(time.monotonic() - t) * 1000:.0f} ms, {rss_mb()} MB",
              file=sys.stderr, flush=True)
        return held
    warming, unheld = loop.run_in_executor(None, warm), False

    async def session(ws):
        nonlocal unheld
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
                msg = msg if isinstance(msg, dict) else {"type": None}
                if msg.get("type") == "hello" and not warming.done():
                    await warming                   # the defaults it asks for are loading already
                await talk.message(msg)
                if msg.get("type") == "hello" and not unheld:
                    unheld = True                   # the first session holds what it uses: the rest may go
                    for kind in warming.result():
                        lib.release(kind, DEFAULTS[kind])
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
    ap.add_argument("--threads", type=int, default=int(os.environ.get("KILN_VOICE_THREADS") or 4))
    a = ap.parse_args()
    asyncio.run(main(a.port, max(1, a.threads)))
