import streamlit as st
from utils.mcp_config import load_mcp_config
from utils.agent_manager import AgentManager

# Setup page and initialize session
st.set_page_config(page_title="Strands Agents", page_icon="🤖", layout="wide")
st.session_state.messages = st.session_state.get("messages", [])

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager()
    with st.spinner("Loading MCP servers..."):
        st.session_state.agent_manager.setup_mcp_clients(load_mcp_config())

st.session_state.agent = st.session_state.get("agent")

def extract_text_content(response):
    """Extract text content from agent response."""
    if not hasattr(response, 'message'):
        return str(response)
        
    msg = response.message
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict) and 'content' in msg:
        content = msg['content']
        if isinstance(content, list):
            return '\n'.join(item.get('text', '') for item in content if isinstance(item, dict))
        if isinstance(content, str):
            return content
    return str(response)

async def generate_stream(prompt):
    async for event in st.session_state.agent.stream_async(prompt):
        if "data" in event:
            yield event["data"]

def create_agent():
    st.session_state.agent = st.session_state.agent_manager.create_agent(
        model_id=st.session_state.get("model_id"),
        temperature=st.session_state.get("temperature", 0.7),
        enable_thinking=st.session_state.get("enable_thinking", False),
        system_prompt=st.session_state.get("system_prompt", "You are a helpful AI assistant.")
    )

def reset_chat():
    st.session_state.messages = []
    create_agent()

# UI Layout
st.title("🤖 Strands Agents")
st.caption("Strands Agents using Amazon Bedrock models")

# Sidebar controls
st.sidebar.button("New Chat", on_click=reset_chat, type="primary", use_container_width=True)

# Model selection
bedrock_models = st.session_state.agent_manager.get_bedrock_models()
model_options = ["Default (Claude 3.7 Sonnet)"] + [f"{name} ({id})" for id, name in bedrock_models]
selected = st.sidebar.selectbox("Model", model_options, key="model_select")
st.session_state.model_id = next((id for id, name in bedrock_models if f"{name} ({id})" == selected), None) if selected != "Default (Claude 3.7 Sonnet)" else None

# Parameters
with st.sidebar.expander("⚙️ Parameters", expanded=False):
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    st.session_state.enable_thinking = st.toggle("Thinking Mode", False) if st.session_state.agent_manager.is_thinking_supported(st.session_state.get("model_id")) else False

# System prompt
with st.sidebar.expander("💬 System Prompt", expanded=False):
    st.session_state.system_prompt = st.text_area("Customize the AI's behavior", value="You are a helpful AI assistant.", height=150)

# MCP Tools
with st.sidebar.expander("🛠️ MCP Tools", expanded=True):
    server_info = st.session_state.agent_manager.get_server_tools_info()
    for server, count in server_info if server_info else []:
        st.info(f"🔌 {server} ({count} tools)")
    if not server_info:
        st.warning("⚠️ No active MCP servers")

# Create agent if needed and display chat
if not st.session_state.agent:
    create_agent()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new messages
if prompt := st.chat_input("Ask something..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Get response (streaming or thinking mode)
            content = extract_text_content(st.session_state.agent(prompt)) if st.session_state.get("enable_thinking") else st.write_stream(generate_stream(prompt))
            st.session_state.messages.append({"role": "assistant", "content": content})
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
