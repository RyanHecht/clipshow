"""Run the web app: python -m clipshow.web [--host 0.0.0.0] [--port 8000] [--share-port 8001]."""

import argparse
import asyncio
import os

import uvicorn


def main() -> None:
    os.environ.setdefault("CLIPSHOW_DEVICE", "cpu")  # query embeddings: fast on CPU, and the GPU is left to workers
    # One short query gains nothing from many threads, and under load they mostly wait on each other.
    threads = os.environ.setdefault("CLIPSHOW_TORCH_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    p = argparse.ArgumentParser(prog="clipshow-web")
    p.add_argument("--host", default=os.environ.get("CLIPSHOW_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("CLIPSHOW_PORT", "8000")))
    p.add_argument("--share-port", type=int, default=int(os.environ.get("CLIPSHOW_SHARE_PORT") or 0),
                   help="also serve share links (and nothing else) on this port, for publishing on its own (0 = off)")
    args = p.parse_args()
    if not args.share_port:
        uvicorn.run("clipshow.web.app:app", host=args.host, port=args.port)
        return
    os.environ["CLIPSHOW_SHARE_PORT"] = str(args.share_port)  # lets the admin page say the listener is on
    servers = [uvicorn.Server(uvicorn.Config(target, host=args.host, port=port))
               for target, port in (("clipshow.web.app:app", args.port), ("clipshow.web.share:public_app", args.share_port))]

    async def serve() -> None:
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)  # one stopping (Ctrl+C, or a port in use) stops both
        for s in servers:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(serve())


if __name__ == "__main__":
    main()
