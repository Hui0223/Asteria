#!/usr/bin/env python3
"""Minimal stdio MCP fixture used by Rust integration tests."""

import json
import sys


def respond(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


for line in sys.stdin:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue

    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        continue

    if method == "initialize":
        requested = request.get("params", {}).get("protocolVersion", "2025-11-25")
        respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": requested,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "asteria-test", "version": "1.0.0"},
                },
            }
        )
    elif method == "tools/list":
        respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo test text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        text = request.get("params", {}).get("arguments", {}).get("text", "")
        respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"echo:{text}"}],
                    "isError": False,
                },
            }
        )
    elif method == "ping":
        respond({"jsonrpc": "2.0", "id": request_id, "result": {}})
    else:
        respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
        )
