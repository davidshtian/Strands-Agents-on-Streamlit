import json
import logging
from pathlib import Path
from typing import Dict, Any


def load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration from default file locations."""
    # Check standard config paths
    paths = [
        Path.home() / ".aws" / "amazonq" / "mcp.json",
        Path.home() / ".mcp.json",
    ]

    # Try to load config from each path
    for path in paths:
        if not path.exists():
            continue

        try:
            # Load the JSON config
            with open(path, "r") as f:
                config = json.load(f)

            # Handle mcpServers format (AmazonQ format)
            if "mcpServers" in config:
                servers = []

                # Convert to the format expected by agent_manager
                for name, server in config["mcpServers"].items():
                    if server.get("disabled", False):
                        continue

                    servers.append(
                        {
                            "name": name,
                            "command": server.get("command", ""),
                            "args": server.get("args", []),
                        }
                    )

                return {"servers": servers}

            # Handle standard format with servers key
            elif "servers" in config:
                return config

            break
        except Exception as e:
            logging.warning(f"Error loading MCP config from {path}: {e}")

    # Return empty config if nothing was found
    return {"servers": []}
