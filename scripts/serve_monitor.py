#!/usr/bin/env python3
"""Serve the static monitor dashboard with optional Basic Auth."""

from __future__ import annotations

import argparse
import base64
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


class AuthenticatedStaticHandler(SimpleHTTPRequestHandler):
    username: str = ""
    password: str = ""

    def do_GET(self) -> None:
        if not self.is_authorized():
            self.send_auth_required()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if not self.is_authorized():
            self.send_auth_required()
            return
        super().do_HEAD()

    def is_authorized(self) -> bool:
        if not self.username and not self.password:
            return True
        header = self.headers.get("Authorization", "")
        prefix = "Basic "
        if not header.startswith(prefix):
            return False
        try:
            decoded = base64.b64decode(header[len(prefix):]).decode("utf-8")
        except Exception:
            return False
        user, sep, password = decoded.partition(":")
        return bool(sep) and user == self.username and password == self.password

    def send_auth_required(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Official V1 Monitor"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Authentication required.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("MONITOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MONITOR_PORT", "8765")))
    parser.add_argument("--directory", default="web/monitor")
    parser.add_argument("--username", default=os.getenv("MONITOR_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("MONITOR_PASSWORD", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = PROJECT_ROOT / args.directory
    if not directory.exists():
        raise SystemExit(f"Monitor directory not found: {directory}")
    if args.host != "127.0.0.1" and (not args.username or not args.password):
        raise SystemExit("Refusing public bind without MONITOR_USERNAME and MONITOR_PASSWORD.")

    AuthenticatedStaticHandler.username = args.username
    AuthenticatedStaticHandler.password = args.password
    handler = partial(AuthenticatedStaticHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    auth_mode = "basic-auth" if args.username or args.password else "no-auth-local"
    print(f"Serving monitor on http://{args.host}:{args.port}/ ({auth_mode})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping monitor server.", file=sys.stderr)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
