"""Stdlib HTTP server exposing the stripe reconstruction API.

Routes
------
* ``GET  /health``                    -- liveness/readiness probe.
* ``POST /api/stripes/reconstruct``   -- stripe reconstruction.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import ApiError, reconstruct_stripe_request

MAX_BODY_BYTES = 8 * 1024 * 1024  # 18 shards * 4096 B base64-encoded << 8 MiB

RECONSTRUCT_PATH = "/api/stripes/reconstruct"
HEALTH_PATH = "/health"


class StripeRequestHandler(BaseHTTPRequestHandler):
    server_version = "StripeReconstructor/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        # Keep access logs on stderr (BaseHTTPRequestHandler default), but
        # route them through a single place for potential silencing.
        super().log_message(fmt, *args)

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path == HEALTH_PATH:
            self._send_json(200, {"status": "ok"})
        else:
            self._error(404, "NOT_FOUND", "unknown route: %s" % self.path)

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path != RECONSTRUCT_PATH:
            self._error(404, "NOT_FOUND", "unknown route: %s" % self.path)
            return

        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length) if content_length is not None else 0
        except ValueError:
            self._error(422, "INVALID_CONTENT_LENGTH", "invalid Content-Length header")
            return
        if length <= 0:
            self._error(422, "INVALID_BODY", "request body must be a non-empty JSON object")
            return
        if length > MAX_BODY_BYTES:
            self._error(413, "PAYLOAD_TOO_LARGE", "request body exceeds %d bytes" % MAX_BODY_BYTES)
            return

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(422, "INVALID_JSON", "request body is not valid JSON")
            return

        try:
            result = reconstruct_stripe_request(body)
        except ApiError as err:
            self._send_json(err.status, err.body())
        except Exception:  # pragma: no cover - last-resort guard
            self.log_message("unhandled error while processing %s", RECONSTRUCT_PATH)
            self._error(500, "INTERNAL_ERROR", "internal error while reconstructing stripe")
        else:
            self._send_json(200, result)

    def do_PUT(self) -> None:  # noqa: N802
        self._error(405, "METHOD_NOT_ALLOWED", "method not allowed")

    def do_DELETE(self) -> None:  # noqa: N802
        self._error(405, "METHOD_NOT_ALLOWED", "method not allowed")


def create_server(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), StripeRequestHandler)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    server = create_server(host, port)
    print("stripe-reconstructor listening on %s:%d" % (host, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
