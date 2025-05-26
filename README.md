# Strands Agents on Streamlit

A streamlined agent application built with [Strands Agents](https://strandsagents.com/) and [Streamlit](https://streamlit.io/), providing a user-friendly interface for interacting with AI agents powered by Amazon Bedrock models, with support for MCP server tools integration.

## Features

- **Multiple Amazon Bedrock Models**: 
  - Uses Claude 3.7 Sonnet as the default model
  - Support for other models like Nova Pro
- **MCP Server Integration**: 
  - Load MCP server configurations from JSON files
  - Extend your agent's capabilities with custom tools
  - Real-time tool execution display
- **Streaming Responses**: 
  - Real-time streaming of agent responses
  - Detailed tool usage information with input/output display

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt

# Recommend UV env
# uv pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

### Configuration

#### Model Configuration

In the sidebar, you can configure the following settings:

- **Model ID** : Specify a custom Amazon Bedrock model ID or leave blank to use the default Claude 3.7
- **Temperature**: Adjust the temperature parameter to control response randomness
- **Enable Thinking Mode**: Toggle to see the agent's detailed reasoning process

#### MCP Configuration

The application automatically loads MCP server configurations from standard locations:

1. The app looks for MCP configuration files in `~/.aws/amazonq/mcp.json` or `~/.mcp.json`
2. Connected MCP servers and available tools are displayed in the "MCP Tools" section of the sidebar
3. Tools from these servers are automatically available to the agent for use during conversations

## Project Structure

```
strands-agents-on-streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── utils/
│   ├── agent_manager.py   # Bedrock model and agent configuration
│   └── mcp_config.py      # MCP configuration loading utilities
└── README.md              # Project documentation
```

## Default MCP Configuration Paths

The application looks for MCP configuration files in the following locations:

1. `~/.mcp.json`
2. `~/.aws/amazonq/mcp.json`

## MCP Configuration Format

The application supports two MCP configuration file formats:

### Format
```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "some-mcp-server-package"],
      "disabled": false
    },
    "another-server": {
      "command": "uv",
      "args": ["run", "another-mcp-server"],
      "env": {
        "VARIABLE": "value"
      }
    }
  }
}
```

## Requirements

- Python 3.10+
- Streamlit
- Strands Agents SDK
- Strands Agents Tools
- MCP