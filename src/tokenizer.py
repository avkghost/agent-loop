"""Token counting utilities for context management."""

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken if available, else fallback to heuristic."""
    # Try tiktoken first
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: simple heuristic (4 chars ≈ 1 token)
    return len(text) // 4
