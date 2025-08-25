
"""
Minimal CSV/XLSX Analysis API with Agentic Chat
- User asks -> ReAct agent (LangChain) -> tools -> generate Python -> run -> return tables/graphics/text
Run:  python3.11 main.py
Docs: http://localhost:8080/docs
"""

import os
import io
import json
import uuid
import math
import shutil
import datetime as dt
from typing import List, Optional, Dict, Any

# Ensure headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np

# Optional .env loader for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Paths ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")
DB_FILE = os.path.join(DATA_DIR, "db.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------- Tiny JSON "DB" ----------
def _load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_FILE):
        return {"files": {}}
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_db(db: Dict[str, Any]) -> None:
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# ---------- Models ----------
class FileInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    path: str
    uploaded_at: str

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    file_ids: List[str]
    # Optional overrides for LLM; defaults are safe
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    max_iterations: int = Field(default=6)
    # Optional: direct code mode (bypass agent)
    mode: str = Field(default="agent", description="agent | python")
    code: Optional[str] = None
    return_artifacts: bool = True

class ChatResponse(BaseModel):
    analysis_id: str
    execution_status: str
    output: str  # final natural-language answer from the agent (or your code)
    tools_used: List[str] = Field(default_factory=list)
    code_used: Optional[str] = None
    stdout: str = ""
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)

# ---------- App ----------
app = FastAPI(title="Agentic CSV/XLSX Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve artifacts and uploads for download
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# ---------- Helpers ----------
def _infer_file_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "xlsx"
    raise HTTPException(status_code=400, detail="Only CSV/XLSX are supported")

def _load_dataframe(file_path: str, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)

def _safe_fig_save(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _new_analysis_dir() -> (str, str):
    analysis_id = uuid.uuid4().hex
    run_dir = os.path.join(ARTIFACT_DIR, analysis_id)
    os.makedirs(run_dir, exist_ok=True)
    return analysis_id, run_dir

# ---------- File Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "time": dt.datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/files/upload")
async def upload_file(file: UploadFile = File(...)):
    file_type = _infer_file_type(file.filename)
    file_id = uuid.uuid4().hex
    ext = ".csv" if file_type == "csv" else ".xlsx"
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(save_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    db = _load_db()
    db["files"][file_id] = {
        "id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "path": save_path,
        "uploaded_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    _save_db(db)

    static_url = f"/static/uploads/{file_id}{ext}"
    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "local_path": save_path,
        "static_url": static_url,
        "message": "File uploaded successfully",
    }

@app.get("/v1/files/{file_id}", response_model=FileInfo)
def get_file(file_id: str):
    db = _load_db()
    info = db["files"].get(file_id)
    if not info:
        raise HTTPException(status_code=404, detail="File not found")
    return info

# ---------- Python Runner (safe, allowlisted imports) ----------
def _run_user_code(code: str, df: pd.DataFrame, run_dir: str):
    artifacts: List[str] = []

    def savefig(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        rel = "/static" + path.replace(DATA_DIR, "").replace("\\\\","/").replace("\\","/")
        artifacts.append(rel)
        return rel

    import io, contextlib, builtins as _builtins
    stdout = io.StringIO()

    allowed_builtins = ["abs","min","max","sum","len","range","print","enumerate","list","dict","set","sorted","any","all","round"]
    safe_builtins = {k: getattr(_builtins, k) for k in allowed_builtins if hasattr(_builtins, k)}

    # Allow scientific libs; block network/process/system-dangerous modules
    def _safe_import(name, *args, **kwargs):
        allowed_prefixes = [
            "pandas", "numpy", "matplotlib", "math", "statistics", "random", "itertools",
            "collections", "datetime", "json", "re", "typing", "os", "pathlib", "sys",
            "scipy", "statsmodels", "sklearn", "pymc", "arviz"
        ]
        banned_prefixes = ["subprocess", "socket", "http", "urllib", "ftplib", "requests", "pexpect", "paramiko"]
        if any(name == b or name.startswith(b + ".") for b in banned_prefixes):
            raise ImportError(f"Import of '{name}' is blocked")
        if any(name == p or name.startswith(p + ".") for p in allowed_prefixes):
            return __import__(name, *args, **kwargs)
        raise ImportError(f"Import of '{name}' is not in allowlist")

    global_env = {"__builtins__": dict(safe_builtins, __import__=_safe_import)}
    local_env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df,
        "RESULTS": [],
        "ARTIFACT_DIR_RUN": run_dir,
        "savefig": savefig,
        "os": os,
        "DATA_DIR": DATA_DIR,
    }

    with contextlib.redirect_stdout(stdout):
        exec(code, global_env, local_env)

    tables = []
    for item in local_env.get("RESULTS", []):
        if isinstance(item, tuple) and len(item) == 2 and hasattr(item[1], "to_dict"):
            name, df_obj = item
            try:
                tables.append({"name": name, "rows": df_obj.head(200).replace({np.nan: None}).to_dict(orient="records")})
            except Exception:
                pass

    return stdout.getvalue(), tables, artifacts

# ---------- LangChain Agent & Tools ----------
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

def _ensure_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set. Put it in .env")

def _validate_file_ids(file_ids: Optional[List[str]]):
    if not file_ids:
        raise HTTPException(status_code=400, detail="Please include 'file_ids' containing at least one uploaded file_id")
    db = _load_db()
    for fid in file_ids:
        if fid not in db["files"]:
            raise HTTPException(status_code=404, detail=f"File not found: {fid}")

def _df_from_file_id(fid: str) -> pd.DataFrame:
    db = _load_db()
    info = db["files"].get(fid)
    if not info:
        raise HTTPException(status_code=404, detail=f"File not found: {fid}")
    return _load_dataframe(info["path"], info["file_type"])

@tool("list_columns", return_direct=False)
def list_columns(json_args: str) -> str:
    """
    List columns and dtypes of a CSV/XLSX.
    Args JSON: {"file_id": "<id>"}
    """
    try:
        args = json.loads(json_args)
        fid = args["file_id"]
        df = _df_from_file_id(fid)
        cols = [{"column": c, "dtype": str(t)} for c, t in zip(df.columns, df.dtypes)]
        return json.dumps({"file_id": fid, "columns": cols})
    except Exception as e:
        return f"ERROR in list_columns: {e}"

@tool("df_head", return_direct=False)
def df_head(json_args: str) -> str:
    """
    Show first N rows of a CSV/XLSX.
    Args JSON: {"file_id": "<id>", "n": 10}
    """
    try:
        args = json.loads(json_args)
        fid = args["file_id"]
        n = int(args.get("n", 10))
        df = _df_from_file_id(fid)
        return json.dumps({
            "file_id": fid,
            "head": df.head(n).replace({np.nan: None}).to_dict(orient="records")
        })
    except Exception as e:
        return f"ERROR in df_head: {e}"

@tool("run_python", return_direct=False)
def run_python(json_args: str) -> str:
    """
    Execute SAFE Python analysis on a CSV/XLSX. You have access to a pandas DataFrame named 'df'.
    Collect DataFrames to return with: RESULTS.append(('name', dataframe))
    Save plots with: out = os.path.join(ARTIFACT_DIR_RUN, 'plot.png'); savefig(out)
    Args JSON: {"file_id":"<id>", "code":"<python code>"}
    """
    try:
        args = json.loads(json_args)
        fid = args["file_id"]
        code = args["code"]
        df = _df_from_file_id(fid)
        analysis_id, run_dir = _new_analysis_dir()
        stdout, tables, artifacts = _run_user_code(code, df, run_dir)
        return json.dumps({
            "analysis_id": analysis_id,
            "code_used": code,
            "stdout": stdout,
            "tables": tables,
            "artifacts": artifacts,
        })
    except Exception as e:
        return f"ERROR in run_python: {e}"

@tool("plot_hist", return_direct=False)
def plot_hist(json_args: str) -> str:
    """
    Draw a histogram for a numeric column and return the artifact URL.
    Args JSON: {"file_id":"<id>", "column":"<col>"}
    """
    try:
        args = json.loads(json_args)
        fid = args["file_id"]
        col = args["column"]
        df = _df_from_file_id(fid)
        if col not in df.columns:
            return f"Column '{col}' not in DataFrame"
        if not np.issubdtype(df[col].dropna().dtype, np.number):
            return f"Column '{col}' is not numeric"
        analysis_id, run_dir = _new_analysis_dir()
        plt.figure(figsize=(8,5))
        df[col].dropna().hist(bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col); plt.ylabel("Frequency")
        out_path = os.path.join(run_dir, f"hist_{col}.png")
        _safe_fig_save(out_path)
        return json.dumps({
            "analysis_id": analysis_id,
            "artifacts": [f"/static/artifacts/{analysis_id}/hist_{col}.png"]
        })
    except Exception as e:
        return f"ERROR in plot_hist: {e}"

LC_TOOLS = [list_columns, df_head, run_python, plot_hist]

def _invoke_react_agent(messages: List[ChatMessage], file_ids: List[str], model: str, temperature: float, max_iterations: int):
    _ensure_openai_key()
    _validate_file_ids(file_ids)
    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, LC_TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=LC_TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True
    )

    file_hint = f"Valid file_ids: {', '.join(file_ids)}. ALWAYS pass a 'file_id' in JSON to tools."

    user_last = messages[-1].content if messages else ""
    agent_input = f"{user_last}\n\n{file_hint}\nIf analysis requires Python, generate the Python and call run_python with it."

    run = executor.invoke({"input": agent_input})
    output = run.get("output", "")
    steps = run.get("intermediate_steps", [])

    tools_used = []
    code_used = None
    stdout = ""
    tables = []
    artifacts = []
    analysis_id = uuid.uuid4().hex

    for (action, observation) in steps:
        tools_used.append(getattr(action, "tool", ""))
        try:
            if isinstance(observation, str) and observation.strip().startswith("{"):
                data = json.loads(observation)
                if "analysis_id" in data:
                    analysis_id = data["analysis_id"]
                if "code_used" in data and isinstance(data["code_used"], str):
                    code_used = data["code_used"]
                if "stdout" in data and isinstance(data["stdout"], str):
                    stdout += (data["stdout"] or "")
                if "tables" in data and isinstance(data["tables"], list):
                    tables.extend(data["tables"])
                if "artifacts" in data and isinstance(data["artifacts"], list):
                    artifacts.extend(data["artifacts"])
        except Exception:
            # ignore JSON parsing errors; some tools may return plain text
            pass

    return {
        "analysis_id": analysis_id,
        "output": output,
        "tools_used": [t for t in tools_used if t],
        "code_used": code_used,
        "stdout": stdout,
        "tables": tables,
        "artifacts": artifacts,
    }

# ---------- Endpoints ----------
@app.post("/v1/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    # Direct python mode (optional bypass)
    if req.mode == "python":
        # Load first file and run provided code
        db = _load_db()
        fid = req.file_ids[0]
        info = db["files"].get(fid)
        if not info:
            raise HTTPException(status_code=404, detail=f"File not found: {fid}")
        df = _load_dataframe(info["path"], info["file_type"])
        analysis_id, run_dir = _new_analysis_dir()
        try:
            stdout, tables, artifacts = _run_user_code(req.code or "", df, run_dir)
            return ChatResponse(
                analysis_id=analysis_id,
                execution_status="completed",
                output="Completed running user-provided Python.",
                tools_used=[],
                code_used=req.code or "",
                stdout=stdout,
                tables=tables,
                artifacts=artifacts if req.return_artifacts else [],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution error: {e}")

    # Default: AGENT mode
    res = _invoke_react_agent(req.messages, req.file_ids, req.model, req.temperature, req.max_iterations)
    return ChatResponse(
        analysis_id=res["analysis_id"],
        execution_status="completed",
        output=res["output"],
        tools_used=res["tools_used"],
        code_used=res["code_used"],
        stdout=res["stdout"],
        tables=res["tables"],
        artifacts=res["artifacts"] if req.return_artifacts else [],
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
