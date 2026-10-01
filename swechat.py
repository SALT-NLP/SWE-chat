"""Helpers for loading SWE-chat and reproducing the paper's figures.

Loads the latest version of the dataset from the Hugging Face Hub. Set the
``SWECHAT_LOCAL_DIR`` environment variable to read from a local copy instead.
"""

from __future__ import annotations

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd

REPO_ID = "SALT-NLP/SWE-chat"

# Repositories whose prompts are machine-generated; excluded from the analyses.
_EXCLUDED_REPOS = ("dayhaysoos/nimbus",)

# Sessions in the release used for the figures in the paper.
_SESSIONS_IN_PAPER = 17_819

VIBE_THRESHOLD = 99
CODING_MODES = ["Human-only", "Collaborative", "Vibe coding"]
MODE_COLORS = dict(zip(CODING_MODES, ["#55a868", "#4c72b0", "#c44e52"]))

INTENT_ORDER = ["create new code", "debug", "refactor", "understand",
                "connect", "test", "git", "other"]
PERSONA_ORDER = ["Expert Nitpicker", "Vague Requester", "Mind Changer", "Other"]

TOOL_ORDER = ["read", "grep", "glob", "bash:file", "bash:build", "bash:net",
              "bash", "git/gh", "write", "edit", "web", "agent", "mcp",
              "TodoWrite", "ToolSearch", "AskUserQuestion", "Skill",
              "EnterPlanMode", "ExitPlanMode", "other"]
TOOL_COLORS = {
    "read": "#154360", "grep": "#7FB3D3", "glob": "#2980B9",
    "bash:file": "#F1948A", "bash:build": "#E74C3C", "bash:net": "#922B21",
    "bash": "#C0392B", "git/gh": "#7B241C", "write": "#1E8449",
    "edit": "#27AE60", "web": "#D35400", "agent": "#7D3C98", "mcp": "#148F77",
    "TodoWrite": "#5D6D7E", "ToolSearch": "#85929E",
    "AskUserQuestion": "#AAB7B8", "Skill": "#2E86C1",
    "EnterPlanMode": "#F39C12", "ExitPlanMode": "#F8C471", "other": "#BDC3C7",
}


def use_paper_style() -> None:
    """Apply the figure styling used throughout the notebook."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.size": 9,
    })


_SHELL_TOOLS = frozenset({"bash", "run_shell_command", "exec_command", "exec",
                          "shell_command", "powershell", "execute"})
_BASH_FILE = frozenset({
    "cd", "ls", "bd", "cat", "mkdir", "rm", "mv", "cp", "find", "touch",
    "chmod", "chown", "head", "tail", "pwd", "echo", "wc", "ln", "stat",
    "du", "df", "tree", "less", "more", "sort", "uniq", "cut", "tee",
    "realpath", "basename", "dirname", "file", "rmdir", "open", "xargs", "lsof"})
_BASH_BUILD = frozenset({
    "go", "cargo", "python", "python3", "uv", "mise", "bun", "npm", "npx",
    "pnpm", "yarn", "xcodebuild", "swift", "make", "rake", "gradle", "mvn",
    "node", "deno", "tsx", "poetry", "pip", "pip3", "pipenv", "rustup",
    "composer", "dart", "flutter", "dotnet", "clang", "clang++", "gcc",
    "g++", "cmake", "ninja", "bazel", "just", "turbo", "nx", "expo",
    "bundle", "pod", "fastlane", "sbt", "ant"})
_BASH_NET = frozenset({
    "curl", "wget", "ssh", "scp", "rsync", "ftp", "sftp", "nc", "netcat",
    "telnet", "ping", "nmap", "dig", "nslookup"})
_RENAMED = {"read_file": "read", "write_file": "write", "multiedit": "edit",
            "apply_patch": "edit", "applypatch": "edit", "replace": "edit",
            "webfetch": "web", "websearch": "web", "todowrite": "TodoWrite",
            "toolsearch": "ToolSearch", "askuserquestion": "AskUserQuestion",
            "skill": "Skill", "enterplanmode": "EnterPlanMode",
            "exitplanmode": "ExitPlanMode"}
_AGENT_TOOLS = frozenset({"task", "taskcreate", "taskget", "taskupdate",
                          "taskoutput", "taskstop", "sendmessage", "tasklist"})


def _table_path(table: str) -> str:
    local = os.environ.get("SWECHAT_LOCAL_DIR")
    if local:
        return os.path.join(local, f"{table}.parquet")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, f"{table}.parquet", repo_type="dataset")


def read_table(table: str, columns: list[str] | None = None,
               exclude_repos: bool = True) -> pd.DataFrame:
    """Read one released table.

    ``exclude_repos`` drops the machine-generated repositories, which the
    paper's analyses leave out. Pass ``False`` for whole-dataset totals.
    """
    if table == "conversations" and columns is None:
        raise ValueError("Pass an explicit columns list when reading conversations; "
                         "the full table is too large to load safely.")
    if exclude_repos and columns is not None and "repo_id" not in columns:
        columns = [*columns, "repo_id"]
    df = pd.read_parquet(_table_path(table), columns=columns)
    if exclude_repos and "repo_id" in df.columns:
        df = df[~df["repo_id"].isin(_EXCLUDED_REPOS)]
    return df.reset_index(drop=True)


def _first_word(command: str) -> str:
    for line in str(command).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            word = line.split()[0]
            return "python" if word.startswith("python") else word
    return ""


def tool_category(tool_name, bash_category=None, command=None) -> str:
    """Map a raw tool name to one of :data:`TOOL_ORDER`."""
    name = str(tool_name).lower().strip() if pd.notna(tool_name) else "other"
    if name in _SHELL_TOOLS:
        word = ""
        if pd.notna(command) and str(command).strip():
            word = _first_word(command)
        elif pd.notna(bash_category) and str(bash_category).lower() == "git":
            word = "git"
        if word in ("git", "gh"):
            return "git/gh"
        if word in ("grep", "rg"):
            return "grep"
        if word in _BASH_FILE:
            return "bash:file"
        if word in _BASH_BUILD:
            return "bash:build"
        if word in _BASH_NET:
            return "bash:net"
        return "bash"
    if name.startswith("mcp__") or name.startswith("mcp_"):
        return "mcp"
    if name in _AGENT_TOOLS:
        return "agent"
    if name in _RENAMED:
        return _RENAMED[name]
    return name if name in TOOL_COLORS else "other"


def load_tool_calls() -> pd.DataFrame:
    """Tool-call turns with a ``tool_category`` column."""
    calls = load_conversations(["turn_type", "tool_name", "bash_category", "command"])
    calls = calls[calls["turn_type"] == "tool_use"].reset_index(drop=True)
    calls["tool_category"] = [
        tool_category(t, b, c) for t, b, c
        in zip(calls["tool_name"], calls["bash_category"], calls["command"])]
    return calls


@lru_cache(maxsize=1)
def session_ids_with_turns() -> frozenset:
    """Session ids that have at least one row in ``conversations``."""
    ids = pd.read_parquet(_table_path("conversations"), columns=["session_id"])
    unique = frozenset(ids["session_id"].unique())
    if len(unique) != _SESSIONS_IN_PAPER:
        print(f"Note: this release has {len(unique):,} sessions; the figures in "
              f"the paper were produced from {_SESSIONS_IN_PAPER:,}.")
    return unique


def coding_mode(sessions: pd.DataFrame) -> pd.Series:
    """Label each session Human-only / Collaborative / Vibe coding."""
    pct = pd.to_numeric(sessions["agent_percentage"], errors="coerce")
    mode = pd.Series("Collaborative", index=sessions.index, dtype="object")
    mode[pct == 0] = "Human-only"
    mode[pct >= VIBE_THRESHOLD] = "Vibe coding"
    mode[pct.isna()] = pd.NA
    return mode


def load_sessions(columns: list[str] | None = None) -> pd.DataFrame:
    """Sessions with a recorded transcript, plus a ``coding_mode`` column."""
    needed = ["session_id", "agent_percentage"]
    if columns is not None:
        columns = list(dict.fromkeys([*needed, *columns]))
    sessions = read_table("sessions", columns)
    sessions = sessions[sessions["session_id"].isin(session_ids_with_turns())]
    sessions = sessions.reset_index(drop=True)
    sessions["coding_mode"] = coding_mode(sessions)
    return sessions


def load_conversations(columns: list[str]) -> pd.DataFrame:
    """Conversation turns. Always pass ``columns`` -- the full table is large."""
    return read_table("conversations", columns)


def load_user_prompts(columns: list[str]) -> pd.DataFrame:
    """User-prompt turns only, streamed so that text columns stay affordable."""
    import pyarrow.parquet as pq

    columns = list(dict.fromkeys([*columns, "turn_type", "repo_id"]))
    keep = []
    for batch in pq.ParquetFile(_table_path("conversations")).iter_batches(
            batch_size=200_000, columns=columns):
        chunk = batch.to_pandas()
        keep.append(chunk[chunk["turn_type"] == "user_prompt"])
    prompts = pd.concat(keep, ignore_index=True)
    return prompts[~prompts["repo_id"].isin(_EXCLUDED_REPOS)].reset_index(drop=True)
