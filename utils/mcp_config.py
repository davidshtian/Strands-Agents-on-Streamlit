import json
import logging
from pathlib import Path
from typing import Dict, Any, List


def load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration from default file locations with improved error handling."""
    logger = logging.getLogger(__name__)

    # Check standard config paths
    paths = [
        Path.home() / ".aws" / "amazonq" / "mcp.json",
        Path.home() / ".mcp.json",
    ]

    # Try to load config from each path
    for path in paths:
        if not path.exists():
            logger.debug(f"Config path does not exist: {path}")
            continue

        try:
            config = _load_config_file(path)
            if config and config.get("servers"):
                logger.info(f"Successfully loaded MCP config from {path}")
                return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
        except IOError as e:
            logger.error(f"Cannot read {path}: {e}")
        except Exception as e:
            logger.warning(f"Error loading MCP config from {path}: {e}")

    logger.info("No valid MCP configuration found")
    return {"servers": []}


def _load_config_file(path: Path) -> Dict[str, Any]:
    """Load and normalize configuration from a single file."""
    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration must be a JSON object")

    # Handle mcpServers format (AmazonQ format)
    if "mcpServers" in raw_config:
        return _normalize_amazonq_format(raw_config["mcpServers"])

    # Handle standard format with servers key
    elif "servers" in raw_config:
        return _validate_standard_format(raw_config)

    else:
        raise ValueError("Configuration must contain 'mcpServers' or 'servers' key")


def _normalize_amazonq_format(mcp_servers: Dict[str, Any]) -> Dict[str, Any]:
    """Convert AmazonQ format to standard format."""
    if not isinstance(mcp_servers, dict):
        raise ValueError("mcpServers must be an object")

    servers = []
    for name, server_config in mcp_servers.items():
        if not isinstance(server_config, dict) or server_config.get("disabled", False):
            continue

        command = server_config.get("command", "")
        args = server_config.get("args", [])

        if command and isinstance(args, list):
            servers.append(
                {
                    "name": name,
                    "command": command,
                    "args": args,
                    "env": server_config.get("env"),
                }
            )

    return {"servers": servers}


def _validate_standard_format(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate standard configuration format."""
    servers = config.get("servers", [])
    if not isinstance(servers, list):
        raise ValueError("'servers' must be a list")

    # Filter out invalid servers
    valid_servers = []
    for server in servers:
        if (
            isinstance(server, dict)
            and server.get("name")
            and server.get("command")
            and isinstance(server.get("args", []), list)
            and not server.get("disabled", False)
        ):
            valid_servers.append(server)

    return {"servers": valid_servers}
