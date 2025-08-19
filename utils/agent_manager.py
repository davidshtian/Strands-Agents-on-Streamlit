import logging
from typing import Dict, Any, Optional, Callable, Tuple, List
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

# Import built-in tools
from strands_tools import (
    agent_graph, calculator, cron, current_time, editor, environment, file_read,
    generate_image, http_request, image_reader, journal, load_tool, nova_reels,
    python_repl, shell, swarm, think, use_aws, workflow
)


class AgentManager:

    # Model configuration
    DEFAULT_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    BEDROCK_MODELS = [("us.amazon.nova-lite-v1:0", "Nova Lite")]
    THINKING_SUPPORTED_MODELS = ["us.anthropic.claude-3-7-sonnet-20250219-v1:0"]

    def __init__(self):
        self.available_tools = []
        self.mcp_clients = {}
        self.server_tools = {}
        self.built_in_tools = self._get_built_in_tools()
        self._current_agent = None

    @classmethod
    def is_thinking_supported(cls, model_id=None):
        return not model_id or model_id in cls.THINKING_SUPPORTED_MODELS

    @classmethod
    def get_bedrock_models(cls):
        return cls.BEDROCK_MODELS

    def setup_mcp_clients(self, mcp_config):
        for server in mcp_config.get("servers", []):
            self._setup_single_mcp_client(server)

    def _setup_single_mcp_client(self, server):
        if not all(k in server for k in ["name", "command"]) or not isinstance(server.get("args", []), list):
            return

        name = server["name"]
        try:
            client = MCPClient(lambda: stdio_client(StdioServerParameters(
                command=server["command"], args=server["args"])))
            self.mcp_clients[name] = client
            
            with client:
                if tools := client.list_tools_sync():
                    self.available_tools.extend(tools)
                    self.server_tools[name] = len(tools)
            client.start()
        except Exception:
            pass


    def get_server_tools_info(self):
        return list(self.server_tools.items())

    def create_model(self, model_id=None, temperature=0.7, enable_thinking=False, thinking_budget_tokens=4096):
        model_params = {"model_id": model_id or self.DEFAULT_MODEL, "temperature": temperature}
        
        if enable_thinking:
            model_params["additional_request_fields"] = {
                "thinking": {"type": "enabled", "budget_tokens": max(1024, thinking_budget_tokens)}
            }
            
        return BedrockModel(**model_params)

    def _get_built_in_tools(self):
        return [agent_graph, calculator, cron, current_time, editor, environment, file_read, 
                generate_image, http_request, image_reader, journal, load_tool, nova_reels, 
                python_repl, shell, swarm, think, use_aws, workflow]

    def create_agent(self, model_id=None, temperature=0.7, system_prompt="You are a helpful AI assistant.", 
                   enable_thinking=False, thinking_budget_tokens=4096, callback_handler=None, custom_tools=None):
        # Create model and collect tools
        model = self.create_model(model_id, temperature, enable_thinking, thinking_budget_tokens)
        all_tools = self.available_tools + self.built_in_tools + (custom_tools or [])
        
        # Create agent
        self._current_agent = Agent(model=model, system_prompt=system_prompt, 
                                   tools=all_tools, callback_handler=callback_handler)
        return self._current_agent

    def shutdown_current_agent(self):
        # Clear agent
        if self._current_agent:
            if hasattr(self._current_agent, "callback_handler"):
                self._current_agent.callback_handler = None
            self._current_agent = None
            
        # Stop clients
        for name, client in list(self.mcp_clients.items()):
            try:
                client.stop(None, None, None)
                self.mcp_clients.pop(name, None)
                self.server_tools.pop(name, None)
            except:
                pass
                
        # Clear tools
        self.available_tools = []
