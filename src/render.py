"""Console rendering and output beautification."""

import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


def _truncate_middle(text: str, limit: int = 200, ellipsis_context: int = 50) -> str:
    """Truncate text in the middle, keeping both start and end visible."""
    if len(text) <= limit:
        return text
    # Show first ~(limit/2 - ellipsis_context/2) chars, ellipsis, last ~(limit/2 - ellipsis_context/2) chars
    start_len = (limit - len("... (truncated, X chars omitted) ...")) // 2
    end_len = start_len
    omitted = len(text) - start_len - end_len
    return f"{text[:start_len]}... (truncated, {omitted} chars omitted) ...{text[-end_len:]}"


def _truncate_value(value, limit: int = 200) -> str:
    """Convert a value to string and truncate if needed."""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)

    if len(text) > limit:
        return _truncate_middle(text, limit)
    return text


def print_banner(goal: str, tool_count: int) -> None:
    """Print startup banner with goal and available tools."""
    console.print(f"\n🚀 Starting agentic loop for goal: '[bold]{goal}[/bold]'\n")
    console.print(Rule(style="cyan"))
    console.print(f"📋 Available tools: {tool_count}")


def print_iteration_header(n: int, max_n: int) -> None:
    """Print iteration separator."""
    console.print(f"\n[cyan]🔄 --- ITERATION {n}/{max_n} ---[/cyan]")


def print_compaction_notice(before: int, after: int) -> None:
    """Print context compaction notice."""
    console.print(f"\n[yellow]⚠️  Context approaching limit ({before} tokens). Compacting history...[/yellow]")
    console.print(f"   [dim]Context size reduced to {after} tokens.[/dim]")


def print_llm_error(msg: str) -> None:
    """Print LM Studio communication error."""
    console.print(f"\n[red]❌ Failed to communicate with LM Studio: {msg}[/red]")


def print_retry_notice(attempt: int, max_attempts: int, wait_time: float) -> None:
    """Print retry notice when LM Studio times out."""
    console.print(f"\n[yellow]⏱️  LM Studio timeout. Retry {attempt}/{max_attempts}, waiting {wait_time:.0f}s...[/yellow]")


def print_empty_response() -> None:
    """Print empty response from LM Studio."""
    console.print("\n[red]❌ Empty response from LM Studio[/red]")


def print_assistant_message(content: str) -> None:
    """Render assistant message with markdown styling."""
    # Check for FINAL_ANSWER prefix
    title = "Assistant"
    if content.startswith("FINAL_ANSWER:"):
        title = "🎯 Final Answer"
        content = content[len("FINAL_ANSWER:"):].lstrip()

    # Render as markdown
    md = Markdown(content)
    panel = Panel(md, title=title, border_style="cyan", expand=False)
    console.print(panel)


def print_tool_call(name: str, args: dict) -> None:
    """Print tool call with arguments, truncating long values for display."""
    console.print(f"\n[bold]🛠️  Calling tool:[/bold] [cyan]{name}[/cyan]")

    if args:
        args_display = {}
        for key, value in args.items():
            # Truncate long string values for display only
            if isinstance(value, str) and len(value) > 200:
                args_display[key] = _truncate_middle(value, limit=200)
            else:
                args_display[key] = value
        console.print(f"   [dim]Arguments:[/dim] {args_display}")


def print_observation(observation_str: str) -> None:
    """
    Parse and display tool observation with proper formatting.
    Handles both JSON envelopes and plain text, with truncation for long content.
    """
    # Try to parse as JSON envelope
    try:
        obs_dict = json.loads(observation_str)
        if isinstance(obs_dict, dict) and "content" in obs_dict:
            content = obs_dict.get("content", "")
            is_error = obs_dict.get("isError", False)
            error_category = obs_dict.get("errorCategory")
            is_retryable = obs_dict.get("isRetryable")

            # Truncate long content for display (but send full to model)
            if len(content) > 2000:
                display_content = _truncate_middle(content, limit=2000, ellipsis_context=100)
            else:
                display_content = content

            # Add metadata annotation if error
            metadata = ""
            if is_error and (error_category or is_retryable is not None):
                parts = []
                if error_category:
                    parts.append(f"errorCategory: {error_category}")
                if is_retryable is not None:
                    parts.append(f"isRetryable: {is_retryable}")
                metadata = f"\n[dim]({', '.join(parts)})[/dim]"

            # Style based on error status
            border_style = "red" if is_error else "green"
            icon = "❌" if is_error else "✅"

            console.print()
            panel = Panel(
                display_content + metadata,
                title=f"{icon} Observation",
                border_style=border_style,
                expand=False
            )
            console.print(panel)
            return
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: plain text observation
    if len(observation_str) > 2000:
        display_str = _truncate_middle(observation_str, limit=2000, ellipsis_context=100)
    else:
        display_str = observation_str

    console.print()
    panel = Panel(display_str, title="✅ Observation", border_style="green", expand=False)
    console.print(panel)


def print_permission_request(kind: str, identifier: str) -> None:
    """Print permission request prompt."""
    console.print("\n" + "="*50)
    console.print("[bold yellow]⚠️  PERMISSION REQUESTED[/bold yellow]")
    console.print(f"[bold]{kind.upper()}:[/bold] {identifier}")


def print_permission_result(label: str) -> None:
    """Print permission result."""
    console.print(f"[green]{label}[/green]")


def print_guardrail_stop() -> None:
    """Print guardrail stop message."""
    console.print("\n[yellow]⚠️  Guardrail: No progress detected. Stopping loop.[/yellow]")


def print_completion() -> None:
    """Print task completion message."""
    console.print("\n[green bold]✅ Task completed successfully![/green bold]")


def print_max_iterations_reached() -> None:
    """Print max iterations reached message."""
    console.print("\n[red bold]🛑 Stopped: Maximum iteration cap reached without completion.[/red bold]")


def print_finish_reason_warning(finish_reason: str) -> None:
    """Print warning for unexpected finish reason."""
    console.print(f"\n[yellow]⚠️  Loop ended with finish_reason: {finish_reason}[/yellow]")
