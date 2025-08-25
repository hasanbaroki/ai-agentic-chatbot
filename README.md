# AI Agentic Chatbot for Data Analysis

A powerful CSV/XLSX analysis API with an agentic chat interface that uses LangChain's ReAct agent to intelligently analyze data files through natural language queries.

## Overview

This application provides an intelligent data analysis system that:
- Accepts CSV and XLSX file uploads
- Processes natural language queries about the data
- Uses a ReAct agent (via LangChain) to determine the best analysis approach
- Executes Python code in a sandboxed environment
- Returns tables, visualizations, and text insights

## Features

### Core Capabilities
- **File Upload & Management**: Upload CSV/XLSX files with automatic storage and retrieval
- **Agentic Chat Interface**: Natural language queries processed by a LangChain ReAct agent
- **Dynamic Code Generation**: Agent generates and executes Python code based on user queries
- **Safe Code Execution**: Sandboxed Python environment with allowlisted imports
- **Multiple Output Formats**: Returns tables, charts, and text analysis
- **Artifact Management**: Automatic storage and serving of generated plots and analysis results

### Agent Tools
The ReAct agent has access to several specialized tools:
1. **list_columns**: Inspect DataFrame columns and data types
2. **df_head**: Preview the first N rows of data
3. **run_python**: Execute custom Python analysis code
4. **plot_hist**: Generate histogram visualizations

## Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key (for the LangChain agent)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-agentic-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
PORT=8080  # Optional, defaults to 8080
```

4. Run the application:
```bash
python3.11 main.py
```

## API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## API Endpoints

### Health Check
```http
GET /health
```
Returns server status and current time.

### File Management

#### Upload File
```http
POST /v1/files/upload
```
Upload a CSV or XLSX file for analysis.

**Request:**
- Body: multipart/form-data with file field

**Response:**
```json
{
  "file_id": "unique_file_id",
  "filename": "data.csv",
  "file_type": "csv",
  "local_path": "/path/to/file",
  "static_url": "/static/uploads/file_id.csv",
  "message": "File uploaded successfully"
}
```

#### Get File Info
```http
GET /v1/files/{file_id}
```
Retrieve metadata about an uploaded file.

### Agent Chat

```http
POST /v1/agent/chat
```
Process natural language queries about uploaded data files.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the average sales by region?"
    }
  ],
  "file_ids": ["file_id_1"],
  "model": "gpt-4o-mini",  // Optional, default: "gpt-4o-mini"
  "temperature": 0.0,       // Optional, default: 0.0
  "max_iterations": 6,      // Optional, default: 6
  "mode": "agent",          // Optional: "agent" or "python"
  "code": null,             // Optional: direct Python code (for "python" mode)
  "return_artifacts": true  // Optional, default: true
}
```

**Response:**
```json
{
  "analysis_id": "unique_analysis_id",
  "execution_status": "completed",
  "output": "Natural language response from the agent",
  "tools_used": ["list_columns", "run_python"],
  "code_used": "# Generated Python code\n...",
  "stdout": "Print output from code execution",
  "tables": [
    {
      "name": "Results",
      "rows": [{"column1": "value1", ...}]
    }
  ],
  "artifacts": ["/static/artifacts/analysis_id/plot.png"]
}
```

## Usage Modes

### 1. Agent Mode (Default)
The LangChain ReAct agent interprets your query, decides which tools to use, and generates appropriate analysis code:

```python
{
  "messages": [{"role": "user", "content": "Show me the distribution of sales"}],
  "file_ids": ["abc123"],
  "mode": "agent"
}
```

### 2. Direct Python Mode
Bypass the agent and run your own Python code directly:

```python
{
  "messages": [{"role": "user", "content": "Custom analysis"}],
  "file_ids": ["abc123"],
  "mode": "python",
  "code": "import pandas as pd\nprint(df.describe())\nRESULTS.append(('Summary', df.describe()))"
}
```

## Code Execution Environment

### Available Variables
- `df`: Pandas DataFrame containing the uploaded file data
- `pd`: Pandas library
- `np`: NumPy library
- `plt`: Matplotlib pyplot
- `RESULTS`: List to collect DataFrames for output
- `ARTIFACT_DIR_RUN`: Directory for saving artifacts
- `savefig(path)`: Helper function to save and register plots

### Allowed Imports
The sandbox allows safe scientific computing libraries:
- pandas, numpy, matplotlib
- math, statistics, random
- itertools, collections, datetime
- json, re, typing
- scipy, statsmodels, sklearn
- pymc, arviz

### Blocked Imports
Network and system-dangerous modules are blocked:
- subprocess, socket, http, urllib
- ftplib, requests, pexpect, paramiko

### Example Analysis Code
```python
# Basic statistics
print(df.describe())
RESULTS.append(('Summary Statistics', df.describe()))

# Groupby analysis
grouped = df.groupby('category')['sales'].sum()
RESULTS.append(('Sales by Category', grouped.to_frame()))

# Visualization
plt.figure(figsize=(10, 6))
df['sales'].hist(bins=30)
plt.title('Sales Distribution')
out_path = os.path.join(ARTIFACT_DIR_RUN, 'sales_dist.png')
savefig(out_path)
```

## Data Storage Structure

```
data/
├── uploads/       # Uploaded CSV/XLSX files
├── artifacts/     # Generated analysis outputs
│   └── {analysis_id}/
│       ├── plot1.png
│       └── plot2.png
└── db.json       # Simple JSON database for file metadata
```

## Static File Access

All uploaded files and generated artifacts are accessible via static URLs:
- Uploads: `/static/uploads/{file_id}.{ext}`
- Artifacts: `/static/artifacts/{analysis_id}/{filename}`

## Security Features

1. **Sandboxed Execution**: Python code runs with restricted builtins and import controls
2. **Allowlisted Imports**: Only safe scientific computing libraries are permitted
3. **Limited File Access**: Code can only write to designated artifact directories
4. **No Network Access**: Network-related modules are blocked
5. **Resource Limits**: Maximum iterations for agent execution

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for LangChain agent functionality
- `PORT`: Server port (default: 8080)

### Request Parameters
- `model`: LLM model to use (default: "gpt-4o-mini")
- `temperature`: LLM temperature for response generation (default: 0.0)
- `max_iterations`: Maximum agent reasoning steps (default: 6)

## Development

### Running Locally
```bash
python3.11 main.py
```
The server will start at `http://localhost:8080`

### CORS Configuration
The API allows all origins by default. Modify the CORS middleware in `main.py` for production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Successful operation
- `400`: Bad request (invalid file type, missing parameters)
- `404`: File not found
- `500`: Server error (execution failure)

Error responses include detailed messages:
```json
{
  "detail": "Only CSV/XLSX are supported"
}
```

## Limitations

1. **File Types**: Only CSV and XLSX files are supported
2. **File Size**: Limited by available memory for pandas operations
3. **Code Execution**: Restricted to allowlisted libraries
4. **Table Output**: Limited to first 200 rows per table
5. **LLM Dependency**: Requires OpenAI API key and internet connection

## Example Use Cases

### 1. Data Exploration
"Show me the first 10 rows and describe the columns"

### 2. Statistical Analysis
"Calculate summary statistics for all numeric columns"

### 3. Data Visualization
"Create a histogram of the age distribution"

### 4. Aggregation
"What's the total sales by region and product category?"

### 5. Correlation Analysis
"Show me the correlation between price and quantity sold"

## Troubleshooting

### OPENAI_API_KEY not set
Ensure your `.env` file contains:
```
OPENAI_API_KEY=sk-...
```

### File upload fails
- Check file format is CSV or XLSX
- Ensure file size is reasonable
- Verify write permissions to `data/` directory

### Agent errors
- Check max_iterations isn't too low
- Verify file_ids are valid
- Ensure query is data-analysis related

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- API Documentation: `http://localhost:8080/docs`
- GitHub Issues: [Add repository issues URL]