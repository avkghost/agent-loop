"""Configuration and global state for the agent loop."""

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "zai-org/glm-4.7-flash"
LM_STUDIO_TIMEOUT = 60  # Request timeout in seconds (LM Studio can be slow)
LM_STUDIO_CONNECT_TIMEOUT = 5  # Connection timeout

# Retry policy for slow/failing LM Studio
LM_STUDIO_RETRIES = 2  # Number of retries on timeout
LM_STUDIO_RETRY_BACKOFF = 2.0  # Backoff multiplier (1st retry: 10s, 2nd: 20s)
LM_STUDIO_RETRY_BASE = 10  # Initial backoff in seconds

# Loop control
MAX_ITERATIONS = 100
MAX_TOKENS = 131072  # Total conversation budget
COMPACT_THRESHOLD = 0.8  # Start compacting at 80% of limit

# Model context window and per-request limits
MODEL_CONTEXT_WINDOW = 131072  # Model's context window size (currently loaded with 131K)
# Common values: 4096 (small), 8192, 16384, 32768, 65536, 131072 (large)
REQUEST_MAX_TOKENS = 8192  # Per-request output limit (increased for 131K context model)
# Formula: min(4096, MODEL_CONTEXT_WINDOW // 3) for auto-calculation
# Set REQUEST_MAX_TOKENS explicitly if you want different behavior

# Tool allowlists and state
ALLOWED_TOOLS = {
    "grep", "wget", "curl", "rg", "cat", "echo", "ls",
    "mkdir", "rm", "mv", "cp", "find", "python", "python3", "gcc", "npm", "ping", "nslookup",
    "timeout", "kill", "jobs", "bg", "fg"  # Shell process control
}
ALLOWED_MCP_TOOLS = set()  # Dynamically populated; starts empty (no implicit trust)

# CLI flags
ALLOW_ALL = False  # Set by main.py based on --allow flag

# Search tracking for implementation-check nudging
SEARCH_COMMANDS = {"grep", "rg", "find"}
HAS_SEARCHED_CODEBASE = False
