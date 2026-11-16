"""Configuration and global state for the agent loop."""

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "zai-org/glm-4.7-flash"
LM_STUDIO_TIMEOUT = 60  # Request timeout in seconds (LM Studio can be slow)
LM_STUDIO_CONNECT_TIMEOUT = 5  # Connection timeout
MCP_TOOL_TIMEOUT = 30  # Maximum time to await an MCP tool response

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
REQUEST_MAX_TOKENS = 8192  # Per-request output limit (increased for 131K context model); YAML: REQUEST_MAX_TOKENS
# Sampling temperature sent with each request. None (default) omits the field, so LM Studio's own setting applies.
# YAML: TEMPERATURE, a number from 0 to 2, or null.
TEMPERATURE = None

# Tool allowlists and state
ALLOWED_TOOLS = {
    "grep", "wget", "curl", "rg", "cat", "echo", "ls",
    "mkdir", "rm", "mv", "cp", "find", "python", "python3", "gcc", "npm", "ping", "nslookup",
    "timeout"
}
ALLOWED_MCP_TOOLS = set()  # Dynamically populated; starts empty (no implicit trust)

# Managed background jobs (src/jobs.py)
JOB_MAX_CONCURRENT = 4                 # running jobs at once; further start_job calls are refused
JOB_DEFAULT_LIFETIME_SECONDS = 600     # watchdog deadline when start_job gives no lifetime_seconds
JOB_MAX_LIFETIME_SECONDS = 3600        # largest lifetime_seconds a model may request
JOB_OUTPUT_BUFFER_BYTES = 1 << 20      # per stream; oldest bytes are dropped and reported
JOB_RETENTION_SECONDS = 600            # finished-job metadata is kept this long
JOB_RETENTION_MAX = 32                 # and at most this many finished jobs are kept
JOB_MAX_RETAINED = 8                   # unconfirmed stops kept for cleanup before new jobs are refused
JOB_GRACE_SECONDS = 5.0                # SIGTERM -> SIGKILL window for a job stop
JOB_KILL_CONFIRM_SECONDS = 5.0         # time to observe the group empty after SIGKILL
JOB_SHUTDOWN_TIMEOUT_SECONDS = 30      # runtime shutdown budget for stopping all jobs

# CLI flags
ALLOW_ALL = False  # Set by main.py based on --allow flag

# Commands that count as a codebase search. Whether a session has searched is session state (authorization.py).
SEARCH_COMMANDS = {"grep", "rg", "find"}
