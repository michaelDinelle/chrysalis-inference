import argparse, asyncio, json, os, signal, sys, threading
from concurrent.futures import ThreadPoolExecutor

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .parts import Library
from .talk import Talk


async def main(port, threads):
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, os._exit, 0)
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
    ap = argparse.ArgumentParser(prog="python -m chrysalis.voice")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CHRYSALIS_VOICE_THREADS") or 4))
    a = ap.parse_args()
    asyncio.run(main(a.port, max(1, a.threads)))
