import json
import logging
from pathlib import Path
from typing import Dict, Any, List


def load_mcp_config() -> Dict[str, Any]:
    """
    Load MCP configuration from default file locations.

    Returns:
        Dict[str, Any]: MCP configuration with server information
    """
    config = {"servers": []}
    config_paths = [
        Path.home() / ".aws" / "amazonq" / "mcp.json",
        Path.home() / ".mcp.json",
    ]

    for config_path in config_paths:
        if not config_path.exists():
            continue

        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            # Convert amazonq format if needed
            if "mcpServers" in loaded_config:
                convert_amazonq_format(loaded_config, config)
            elif "servers" in loaded_config:
                config = loaded_config

            # Successfully loaded, no need to check other paths
            break
        except Exception as e:
            logging.warning(f"Error loading MCP config from {config_path}: {e}")

    return config


def convert_amazonq_format(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    """
    Convert amazonq MCP format to standard format.

    Args:
        source: Source config in amazonq format
        target: Target config to store converted data
    """
    for name, server_config in source["mcpServers"].items():
        # Skip disabled servers
        if server_config.get("disabled", False):
            continue

        # Add to target config in standard format
        target["servers"].append(
            {
                "name": name,
                "command": server_config.get("command", ""),
                "args": server_config.get("args", []),
            }
        )
