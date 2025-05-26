import json
import streamlit as st
from utils.mcp_config import load_mcp_config
from utils.agent_manager import AgentManager

# Set up page configuration
st.set_page_config(page_title="Strands Agents", page_icon="🤖", layout="wide")

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.agent_manager = AgentManager()
    st.session_state.agent = None
    st.session_state.tool_calls = {}
    st.session_state.displayed_tools = set()
    st.session_state.current_message_tools = {}
    st.session_state.mcp_loading = True
    st.session_state.mcp_loaded = False
    # Start initial MCP client setup
    with st.spinner("Loading MCP servers..."):
        st.session_state.agent_manager.setup_mcp_clients(load_mcp_config())
        st.session_state.mcp_loaded = bool(
            st.session_state.agent_manager.get_server_tools_info()
        )
        st.session_state.mcp_loading = False


def handle_streaming_data(data):
    """Handle streaming text data from the agent."""
    st.session_state.current_response += data
    if "response_container" in st.session_state:
        st.session_state.response_container.markdown(st.session_state.current_response)


def handle_tool_result(message):
    """Handle tool results from agent messages."""
    if (
        not isinstance(message, dict)
        or message.get("role") != "user"
        or not message.get("content")
    ):
        return

    for content_item in message["content"]:
        if isinstance(content_item, dict) and "toolResult" in content_item:
            tool_result = content_item["toolResult"]
            tool_id = tool_result.get("toolUseId", "unknown")

            if tool_id in st.session_state.tool_calls:
                result_content = next(
                    (
                        content["text"]
                        for content in tool_result.get("content", [])
                        if isinstance(content, dict) and content.get("text")
                    ),
                    None,
                )

                if result_content:
                    result_info = (
                        f"\n⚡ **Raw Tool Result:**\n```json\n{result_content}\n```\n"
                    )
                    result_info += "--------------------\n"
                    update_response(result_info)


def handle_tool_use(tool_use):
    """Handle tool use information."""
    tool_id = tool_use.get("toolUseId", "unknown")
    tool_name = tool_use.get("name", "")

    # Skip if this tool has already been displayed
    if not tool_name or tool_name in st.session_state.displayed_tools:
        return

    st.session_state.displayed_tools.add(tool_name)

    # Store tool call data in both global and current message contexts
    tool_data = {
        "name": tool_name,
        "input": tool_use.get("input", {}),
        "output": None,
        "status": "started",
    }

    st.session_state.tool_calls[tool_id] = tool_data
    st.session_state.current_message_tools[tool_id] = tool_data

    # Format and add tool info to response
    tool_info = f"\n\n🛠️ **Tool Call: `{tool_name}`**\n"

    # Add formatted input parameters
    if tool_use.get("input"):
        tool_info += "📥 **Input:**\n```json\n"
        tool_info += json.dumps(tool_use.get("input"), indent=2)
        tool_info += "\n```\n"
    else:
        tool_info += "📥 **Input:** No parameters\n"

    update_response(tool_info)


def handle_tool_output(tool_use):
    """Handle tool output or errors."""
    tool_id = tool_use.get("toolUseId", "unknown")
    if tool_id not in st.session_state.tool_calls:
        return

    tool_name = tool_use.get("name", "")

    # Handle successful output
    if tool_use.get("output"):
        output_data = tool_use.get("output")

        # Update both global and current message tool data
        st.session_state.tool_calls[tool_id]["output"] = output_data
        st.session_state.tool_calls[tool_id]["status"] = "completed"

        if tool_id in st.session_state.current_message_tools:
            st.session_state.current_message_tools[tool_id]["output"] = output_data
            st.session_state.current_message_tools[tool_id]["status"] = "completed"

        output_str = str(output_data)
        output_display = output_str[:500] + ("..." if len(output_str) > 500 else "")

        output_info = (
            f"\n📤 **Result from `{tool_name}`:**\n```\n{output_display}\n```\n"
        )
        output_info += "--------------------\n"
        update_response(output_info)

    # Handle errors
    elif tool_use.get("error"):
        error_data = tool_use.get("error")

        # Update both global and current message tool data
        st.session_state.tool_calls[tool_id]["error"] = error_data
        st.session_state.tool_calls[tool_id]["status"] = "failed"

        if tool_id in st.session_state.current_message_tools:
            st.session_state.current_message_tools[tool_id]["error"] = error_data
            st.session_state.current_message_tools[tool_id]["status"] = "failed"

        error_info = f"\n❌ **Error from `{tool_name}`:**\n```\n{error_data}\n```\n"
        error_info += "--------------------\n"
        update_response(error_info)


def update_response(content):
    """Add content to the current response and update the UI."""
    st.session_state.current_response += content
    if "response_container" in st.session_state:
        st.session_state.response_container.markdown(st.session_state.current_response)


def chat_callback_handler(**kwargs):
    """Callback handler for streaming agent responses."""
    if "data" in kwargs:
        handle_streaming_data(kwargs["data"])
    elif "message" in kwargs and isinstance(kwargs["message"], dict):
        handle_tool_result(kwargs["message"])
    elif "current_tool_use" in kwargs:
        tool_use = kwargs["current_tool_use"]

        # Handle tool use events
        if tool_use.get("name"):
            handle_tool_use(tool_use)

        # Handle tool output/error events
        if tool_use.get("output") or tool_use.get("error"):
            handle_tool_output(tool_use)


def create_agent_with_config():
    """Create a new agent based on the current configuration."""
    model_id = st.session_state.get("model_id", None)
    is_thinking_supported = st.session_state.agent_manager.is_thinking_supported(
        model_id
    )

    # Configuration parameters
    config = {
        "model_id": model_id,
        "temperature": st.session_state.get("temperature", 0.7),
        "enable_thinking": (
            st.session_state.get("enable_thinking", False)
            if is_thinking_supported
            else False
        ),
        "thinking_budget_tokens": st.session_state.get("thinking_budget_tokens", 4096),
        "callback_handler": (
            None
            if st.session_state.get("enable_thinking", False)
            else chat_callback_handler
        ),
    }

    st.session_state.agent = st.session_state.agent_manager.create_agent(**config)
    st.session_state.initialized = True
    return st.session_state.agent


def reset_agent():
    """Reset the agent and clear the chat history."""
    # First, shutdown any existing agent and MCP clients
    if st.session_state.agent_manager:
        # Set loading state
        st.session_state.mcp_loading = True
        st.session_state.mcp_loaded = False

        # Shutdown current agent
        st.session_state.agent_manager.shutdown_current_agent()

        # Show spinner during MCP loading
        with st.spinner("Loading MCP servers..."):
            # Reestablish MCP connections after shutdown
            st.session_state.agent_manager.setup_mcp_clients(load_mcp_config())
            st.session_state.mcp_loaded = bool(
                st.session_state.agent_manager.get_server_tools_info()
            )
            st.session_state.mcp_loading = False

            # No success message as per user feedback

    # Reset UI state
    st.session_state.messages = []
    st.session_state.tool_calls = {}
    st.session_state.displayed_tools = set()
    st.session_state.current_message_tools = {}

    # Create a new agent
    create_agent_with_config()


# Main UI
st.title("🤖 Strands Agents")
st.caption("Strands Agents using Amazon Bedrock models")

# Sidebar configuration
st.sidebar.button(
    "New Chat", on_click=reset_agent, type="primary", use_container_width=True
)

# Model selection
bedrock_models = st.session_state.agent_manager.get_bedrock_models()
model_options = ["Default (Claude 3.7 Sonnet)"] + [
    f"{display_name} ({model_id})" for model_id, display_name in bedrock_models
]


def update_model_selection():
    """Update model ID in session state and create a new agent."""
    selected_option = st.session_state.selected_model_option
    # Extract model_id from selection
    model_id_selected = (
        ""  # Use default
        if selected_option == "Default (Claude 3.7 Sonnet)"
        else next(
            (
                model_id
                for model_id, display_name in bedrock_models
                if f"{display_name} ({model_id})" == selected_option
            ),
            "",
        )
    )
    st.session_state.model_id = model_id_selected
    # Force agent recreation with the new model
    create_agent_with_config()


with st.sidebar.expander("📋 Model Selection", expanded=True):
    st.selectbox(
        "Model",
        options=model_options,
        index=0,
        key="selected_model_option",
        on_change=update_model_selection,
    )

# Check if thinking mode is supported
is_thinking_supported = st.session_state.agent_manager.is_thinking_supported(
    st.session_state.get("model_id", None)
)

# Parameters configuration
with st.sidebar.expander("⚙️ Parameters", expanded=False):
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="temperature",
        on_change=create_agent_with_config,
    )

    # Show thinking mode toggle only for supported models
    if is_thinking_supported:
        # Reset thinking mode if changing to a model that doesn't support it
        if st.session_state.get("enable_thinking", False) and not is_thinking_supported:
            st.session_state.enable_thinking = False

        st.toggle(
            "Thinking Mode",
            value=st.session_state.get("enable_thinking", False),
            key="enable_thinking",
            help="Show agent reasoning process",
            on_change=create_agent_with_config,
        )

        if st.session_state.get("enable_thinking", False):
            st.slider(
                "Thinking Budget",
                min_value=1024,
                max_value=32768,
                value=4096,
                step=1024,
                key="thinking_budget_tokens",
                help="Token budget for reasoning",
                on_change=create_agent_with_config,
            )
    else:
        st.session_state.enable_thinking = False

# MCP configuration
with st.sidebar.expander("🛠️ MCP Tools", expanded=True):
    # Display loading status
    if st.session_state.mcp_loading:
        st.spinner("Loading MCP servers...")
        st.info("MCP servers are initializing...")
    else:
        # Show server info without the success message
        server_info = st.session_state.agent_manager.get_server_tools_info()
        if server_info:
            for server, tool_count in server_info:
                st.info(f"🔌 {server} ({tool_count} tools)")
        else:
            st.warning("⚠️ No active MCP servers")

# Initialize agent if needed
if not st.session_state.initialized:
    create_agent_with_config()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat functionality
if st.session_state.mcp_loading:
    # Show loading message with prominent spinner
    loading_col1, loading_col2 = st.columns([1, 4])
    with loading_col1:
        st.spinner()
    with loading_col2:
        st.info(
            "Loading MCP servers... The chat will be available once servers are connected."
        )

    # Disable chat input during loading
    st.chat_input("Loading servers...", disabled=True)
elif st.session_state.mcp_loaded:
    # Enable chat input when MCP servers are loaded
    user_input = st.chat_input("Ask something...")
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            st.session_state.response_container = response_placeholder
            st.session_state.current_response = ""
            st.session_state.displayed_tools = set()

            try:
                response = st.session_state.agent(user_input)

                # Handle response based on thinking mode
                response_content = (
                    response.message
                    if st.session_state.enable_thinking
                    else st.session_state.current_response
                )

                # Save assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_content}
                )

                if st.session_state.enable_thinking:
                    response_placeholder.markdown(response_content)

            except Exception as e:
                error_message = f"Error: {str(e)}"
                response_placeholder.error(error_message)
                response_content = error_message

                # Save error message to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

            # Store the current message tools with the response
            tool_content = ""
            if (
                st.session_state.current_message_tools
                and not st.session_state.enable_thinking
            ):
                tool_content = "\n\n**Tool Calls:**\n"
                for (
                    tool_id,
                    tool_data,
                ) in st.session_state.current_message_tools.items():
                    tool_content += f"- **{tool_data['name']}**\n"
                    if tool_data.get("input"):
                        tool_content += (
                            f"  - Input: `{json.dumps(tool_data['input'])}`\n"
                        )
                    if tool_data.get("output"):
                        output_str = str(tool_data["output"])
                        output_display = output_str[:200] + (
                            "..." if len(output_str) > 200 else ""
                        )
                        tool_content += f"  - Output: `{output_display}`\n"
                    if tool_data.get("error"):
                        tool_content += f"  - Error: `{tool_data['error']}`\n"

            # Reset current message tools for next interaction
            st.session_state.current_message_tools = {}
else:
    # No MCP servers loaded
    st.error("❌ No MCP servers available. Chat functionality is disabled.")

    # Retry button for server connection
    if st.button("Retry MCP Connection", type="primary", use_container_width=True):
        # Set loading state
        st.session_state.mcp_loading = True

        # Show spinner during MCP loading attempt
        with st.spinner("Attempting to connect to MCP servers..."):
            st.session_state.agent_manager.setup_mcp_clients(load_mcp_config())
            st.session_state.mcp_loaded = bool(
                st.session_state.agent_manager.get_server_tools_info()
            )
            st.session_state.mcp_loading = False

            # Show toast notification only for failure
            if st.session_state.mcp_loaded:
                # Just rerun without the success message
                st.rerun()
            else:
                st.toast("Failed to connect to MCP servers", icon="❌")

    # Disable chat input
    st.chat_input("MCP servers unavailable", disabled=True)
