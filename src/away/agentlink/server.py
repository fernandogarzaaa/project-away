"""Runs the Agent Link ASGI app with uvicorn on 127.0.0.1 in a background thread (never on another interface)."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

HOST = "127.0.0.1"


class Server:
    def __init__(self, app: Any, port: int = 0) -> None:
        self.app = app
        self.requested_port = int(port)
        self.port = 0
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 15.0) -> int:
        import uvicorn

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, self.requested_port))
        self.port = int(sock.getsockname()[1])
        config = uvicorn.Config(self.app, log_level="warning", access_log=False, lifespan="on", timeout_keep_alive=30)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [sock]}, daemon=True, name="agentlink-server"
        )
        self._thread.start()
        end = time.monotonic() + timeout
        while not getattr(self._server, "started", False):
            if time.monotonic() > end or not self._thread.is_alive():
                raise RuntimeError("the Agent Link server did not start")
            time.sleep(0.05)
        return self.port

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        self._server = None
        self._thread = None
