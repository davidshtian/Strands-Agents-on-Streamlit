import logging
from typing import Dict, Any, Optional, Callable, Tuple, List
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

# Import built-in tools grouped by categories
from strands_tools import (
    # Agent tools
    agent_graph,
    swarm,
    workflow,
    # Computation tools
    calculator,
    python_repl,
    think,
    # System tools
    cron,
    current_time,
    environment,
    shell,
    # File handling tools
    editor,
    file_read,  # retrieve,
    # Media tools
    generate_image,
    image_reader,
    nova_reels,
    # Web and external services
    http_request,
    journal,
    load_tool,
    use_aws,  # use_llm,
)


class AgentManager:
    """
    Manages the creation and configuration of Strands Agents with Bedrock models.

    This class handles model configuration, tool integration, and MCP client setup
    for Strands Agent interactions with Amazon Bedrock.
    """

    # Model configuration
    DEFAULT_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    BEDROCK_MODELS = [("us.amazon.nova-lite-v1:0", "Nova Lite")]
    THINKING_SUPPORTED_MODELS = ["us.anthropic.claude-3-7-sonnet-20250219-v1:0"]

    def __init__(self):
        """Initialize the AgentManager with empty collections for tools and clients."""
        self.available_tools = []
        self.mcp_clients = {}
        self.server_tools = {}
        self.built_in_tools = self._get_built_in_tools()
        self._current_agent = None

    @classmethod
    def is_thinking_supported(cls, model_id: Optional[str] = None) -> bool:
        """Check if the specified model supports thinking mode."""
        return True if not model_id else model_id in cls.THINKING_SUPPORTED_MODELS

    @classmethod
    def get_bedrock_models(cls) -> List[Tuple[str, str]]:
        """Get the list of available Bedrock models."""
        return cls.BEDROCK_MODELS

    def setup_mcp_clients(self, mcp_config: Dict[str, Any]) -> None:
        """Set up MCP clients from the provided configuration."""
        if not mcp_config.get("servers"):
            return

        for server in mcp_config["servers"]:
            self._setup_single_mcp_client(server)

    def _setup_single_mcp_client(self, server: Dict[str, Any]) -> None:
        """Set up a single MCP client and load its tools."""
        # Validate server configuration
        if not isinstance(server, dict) or "name" not in server:
            return

        name = server["name"]
        if "command" not in server or "args" not in server:
            logging.error(f"Missing 'command' or 'args' for MCP server {name}")
            return

        try:
            # Create and connect client
            client = MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command=server["command"], args=server["args"]
                    )
                )
            )
            self.mcp_clients[name] = client

            # Connect and load tools in one block
            with client:
                if tools := client.list_tools_sync():
                    self.available_tools.extend(tools)
                    self.server_tools[name] = len(tools)
            client.start()
        except Exception as e:
            logging.error(f"Failed to setup MCP client for {name}: {e}")

    def get_server_tools_info(self) -> List[Tuple[str, int]]:
        """Get information about servers and their tool counts."""
        return [(name, count) for name, count in self.server_tools.items()]

    def create_model(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 4096,
    ) -> BedrockModel:
        """Create a Bedrock model with specified parameters."""
        # Set model parameters
        model_id = model_id or self.DEFAULT_MODEL
        model_params = {"model_id": model_id, "temperature": temperature}

        # Configure thinking mode if enabled
        if enable_thinking:
            model_params["additional_request_fields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": max(1024, thinking_budget_tokens),
                }
            }

        return BedrockModel(**model_params)

    def _get_built_in_tools(self) -> List:
        """Get a list of all built-in tools from strands-agents-tools."""
        return [
            # Agent tools
            agent_graph,
            calculator,
            cron,
            current_time,
            editor,
            environment,
            file_read,
            generate_image,
            http_request,
            image_reader,
            journal,
            load_tool,
            nova_reels,
            python_repl,
            shell,
            swarm,
            think,
            use_aws,
            workflow,
            # Commented out tools
            # retrieve, use_llm
        ]

    def create_agent(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful AI assistant.",
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 4096,
        callback_handler: Optional[Callable] = None,
        custom_tools: Optional[List] = None,
    ) -> Agent:
        """Create and configure a Strands Agent with specified settings."""
        # Create model
        model = self.create_model(
            model_id=model_id,
            temperature=temperature,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        # Configure logging for thinking mode
        if enable_thinking:
            logging.getLogger("strands").setLevel(logging.DEBUG)
            logging.basicConfig(
                format="%(levelname)s | %(name)s | %(message)s",
                handlers=[logging.StreamHandler()],
            )

        # Collect all tools
        all_tools = self.available_tools + self.built_in_tools
        if custom_tools:
            all_tools.extend(custom_tools)

        # Create and store agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=all_tools,
            callback_handler=callback_handler,
        )
        self._current_agent = agent

        return agent

    def shutdown_current_agent(self) -> None:
        """Shutdown the current agent and its MCP clients."""
        # Terminate agent if it exists
        if self._current_agent:
            try:
                # Clear any ongoing callbacks to interrupt active operations
                if hasattr(self._current_agent, "callback_handler"):
                    self._current_agent.callback_handler = None

                # Log the termination of model resources if they exist
                if hasattr(self._current_agent, "model") and self._current_agent.model:
                    logging.info("Terminating model client resources")
            except Exception as e:
                logging.error(f"Error while terminating agent: {e}")

            # Always reset agent reference regardless of errors
            self._current_agent = None
            logging.info("Agent reference cleared")

        # Cleanup MCP clients
        for name, client in list(self.mcp_clients.items()):
            try:
                logging.info(f"Stopping MCP client: {name}")
                client.stop(None, None, None)

                # Remove references and log success
                self.mcp_clients.pop(name, None)
                self.server_tools.pop(name, None)
                logging.info(f"MCP client {name} stopped successfully")
            except Exception as e:
                logging.error(f"Error shutting down MCP client {name}: {e}")

        # Reset available tools
        self.available_tools = []
        logging.info("Agent and MCP clients shutdown complete")
