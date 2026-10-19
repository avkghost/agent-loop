"""Load agents from markdown files."""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import yaml


@dataclass
class Agent:
    """Represents an agent configuration."""
    name: str
    model: str
    temperature: float
    max_tokens: int
    tools: list
    system_prompt: str


def load_agents(agents_dir: str = "agents") -> Dict[str, Agent]:
    """Load all agents from markdown files in the agents directory."""
    agents = {}
    agents_path = Path(agents_dir)

    if not agents_path.exists():
        return agents

    for md_file in sorted(agents_path.glob("*.md")):
        agent_name = md_file.stem
        try:
            agent = parse_agent_file(md_file)
            if agent:
                agents[agent_name] = agent
        except Exception as e:
            print(f"⚠️  Warning: Failed to load agent '{agent_name}': {e}")

    return agents


def parse_agent_file(filepath: Path) -> Optional[Agent]:
    """Parse a single agent markdown file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract YAML frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not frontmatter_match:
        return None

    try:
        frontmatter_text = frontmatter_match.group(1)
        system_prompt = frontmatter_match.group(2).strip()

        # Parse YAML
        frontmatter = yaml.safe_load(frontmatter_text) or {}

        # Extract fields
        name = filepath.stem
        model = frontmatter.get('model', 'default-model')
        temperature = float(frontmatter.get('temperature', 0.7))
        max_tokens = int(frontmatter.get('max_tokens', 4096))
        tools = frontmatter.get('tools', []) or []

        return Agent(
            name=name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system_prompt=system_prompt
        )
    except Exception as e:
        print(f"⚠️  Error parsing agent file {filepath}: {e}")
        return None


def get_agent_description(agent: Agent) -> str:
    """Extract first line of system prompt as description."""
    first_line = agent.system_prompt.split('\n')[0]
    return first_line.strip() if first_line else "(No description)"
