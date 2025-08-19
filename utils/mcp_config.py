import json
from pathlib import Path


def load_mcp_config():
    # Try standard config paths
    for path in [Path.home() / ".aws" / "amazonq" / "mcp.json", Path.home() / ".mcp.json"]:
        if not path.exists():
            continue
            
        try:
            config = _load_config_file(path)
            if config and config.get("servers"):
                return config
        except:
            pass
            
    return {"servers": []}

def _load_config_file(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    if not isinstance(config, dict):
        return {"servers": []}
        
    # Handle different formats
    if "mcpServers" in config:
        return _normalize_amazonq_format(config["mcpServers"])
    elif "servers" in config:
        return _validate_standard_format(config)
    
    return {"servers": []}

def _normalize_amazonq_format(mcp_servers):
    if not isinstance(mcp_servers, dict):
        return {"servers": []}
        
    servers = []
    for name, cfg in mcp_servers.items():
        if isinstance(cfg, dict) and not cfg.get("disabled", False):
            command = cfg.get("command", "")
            args = cfg.get("args", [])
            
            if command and isinstance(args, list):
                servers.append({
                    "name": name,
                    "command": command,
                    "args": args,
                    "env": cfg.get("env")
                })
                
    return {"servers": servers}

def _validate_standard_format(config):
    servers = config.get("servers", [])
    if not isinstance(servers, list):
        return {"servers": []}
        
    return {"servers": [s for s in servers if isinstance(s, dict) and 
                      s.get("name") and s.get("command") and 
                      isinstance(s.get("args", []), list) and 
                      not s.get("disabled", False)]}
