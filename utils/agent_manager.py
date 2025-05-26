import logging
from typing import Dict, Any, Optional, Callable, Tuple, List
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

# Import built-in tools - organized by categories
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
    file_read,
    retrieve,
    # Media tools
    generate_image,
    image_reader,
    nova_reels,
    # Web and external services
    http_request,
    journal,
    load_tool,
    use_aws,
    use_llm,
)


class AgentManager:
    """
    Manages the creation and configuration of Strands Agents with Bedrock models.

    This class handles model configuration, tool integration, and MCP client setup
    for Strands Agent interactions with Amazon Bedrock.
    """

    # Default model and model options
    DEFAULT_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    BEDROCK_MODELS = [
        ("us.amazon.nova-lite-v1:0", "Nova Lite"),
    ]

    # Models that support thinking mode
    THINKING_SUPPORTED_MODELS = [
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ]

    def __init__(self):
        """Initialize the AgentManager with empty collections for tools and clients."""
        self.available_tools = []
        self.mcp_clients = {}
        self.server_tools = {}
        self.built_in_tools = self._get_built_in_tools()

    @classmethod
    def is_thinking_supported(cls, model_id: Optional[str] = None) -> bool:
        """
        Check if the specified model supports thinking mode.

        Args:
            model_id: The model ID to check for thinking mode support

        Returns:
            bool: True if thinking is supported, False otherwise
        """
        # Default model supports thinking
        if not model_id:
            return True
        return model_id in cls.THINKING_SUPPORTED_MODELS

    @classmethod
    def get_bedrock_models(cls) -> List[Tuple[str, str]]:
        """
        Get the list of available Bedrock models.

        Returns:
            List[Tuple[str, str]]: List of (model_id, display_name) tuples
        """
        return cls.BEDROCK_MODELS

    def setup_mcp_clients(self, mcp_config: Dict[str, Any]) -> None:
        """
        Set up MCP clients from the provided configuration.

        Args:
            mcp_config: MCP configuration dictionary with server details
        """
        if not mcp_config.get("servers"):
            return

        for server in mcp_config["servers"]:
            self._setup_single_mcp_client(server)

    def _setup_single_mcp_client(self, server: Dict[str, Any]) -> None:
        """
        Set up a single MCP client and load its tools.

        Args:
            server: Server configuration dictionary
        """
        if not isinstance(server, dict) or "name" not in server:
            return

        name = server["name"]
        if "command" not in server or "args" not in server:
            logging.error(f"Missing 'command' or 'args' for MCP server {name}")
            return

        try:
            # Create and configure MCP client
            client = MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command=server["command"], args=server["args"]
                    )
                )
            )
            self.mcp_clients[name] = client

            # Connect and load tools
            with client:
                tools = client.list_tools_sync()
                if tools:
                    self.available_tools.extend(tools)
                    self.server_tools[name] = len(tools)
            client.start()
        except Exception as e:
            logging.error(f"Failed to setup MCP client for {name}: {e}")

    def get_server_tools_info(self) -> List[Tuple[str, int]]:
        """
        Get information about servers and their tool counts.

        Returns:
            List[Tuple[str, int]]: List of (server_name, tool_count) tuples
        """
        return [(name, count) for name, count in self.server_tools.items()]

    def create_model(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 4096,
    ) -> BedrockModel:
        """
        Create a Bedrock model object with the specified parameters.

        Args:
            model_id: The model ID to use (uses default if None)
            temperature: Model temperature parameter (0.0-1.0)
            enable_thinking: Whether to enable thinking mode
            thinking_budget_tokens: Token budget for thinking mode

        Returns:
            BedrockModel: Configured model instance
        """
        model_id = model_id or self.DEFAULT_MODEL
        model_params = {"model_id": model_id, "temperature": temperature}

        # Add thinking configuration if enabled
        if enable_thinking:
            model_params["additional_request_fields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": max(1024, thinking_budget_tokens),
                }
            }

        return BedrockModel(**model_params)

    def _get_built_in_tools(self) -> List:
        """
        Get a list of all built-in tools from strands-agents-tools.

        Returns:
            List: Available built-in tools
        """
        return [
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
            retrieve,
            shell,
            swarm,
            think,
            use_aws,
            use_llm,
            workflow,
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
        """
        Create and configure a Strands Agent with the specified settings.

        Args:
            model_id: The model ID to use
            temperature: Model temperature parameter (0.0-1.0)
            system_prompt: The system prompt for the agent
            enable_thinking: Whether to enable thinking mode
            thinking_budget_tokens: Token budget for thinking mode
            callback_handler: Callback handler for streaming responses
            custom_tools: Additional custom tools to include

        Returns:
            Agent: Configured Strands Agent instance
        """
        # Create model with appropriate configuration
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

        # Combine all tools (built-in, MCP, and custom)
        all_tools = self.available_tools + self.built_in_tools
        if custom_tools:
            all_tools.extend(custom_tools)

        return Agent(
            model=model,
            system_prompt=system_prompt,
            tools=all_tools,
            callback_handler=callback_handler,
        )
