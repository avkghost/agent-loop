"""Load skills from markdown files."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml


@dataclass
class Skill:
    """Represents a skill configuration."""
    name: str
    description: str
    allowed_tools: Optional[list] = None
    system_prompt: str = ""
    scripts: Optional[Dict[str, Path]] = None
    references: Optional[Dict[str, Path]] = None
    assets: Optional[Dict[str, Path]] = None


def load_skills(skills_dir: str = "skills") -> Dict[str, Skill]:
    """Load all skills from markdown files in the skills directory."""
    skills = {}
    skills_path = Path(skills_dir)

    if not skills_path.exists():
        return skills

    # Load directory-form skills first (skills/<name>/SKILL.md)
    for dir_entry in sorted(skills_path.iterdir()):
        if dir_entry.is_dir():
            skill_md = dir_entry / "SKILL.md"
            if skill_md.exists():
                skill_name = dir_entry.name
                try:
                    skill = parse_skill_directory(dir_entry)
                    if skill:
                        skills[skill_name] = skill
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load skill '{skill_name}': {e}")

    # Load flat-form skills (skills/<name>.md), skipping if directory form was found
    for md_file in sorted(skills_path.glob("*.md")):
        skill_name = md_file.stem
        if skill_name not in skills:  # Don't override directory-form
            try:
                skill = parse_skill_file(md_file)
                if skill:
                    skills[skill_name] = skill
            except Exception as e:
                print(f"⚠️  Warning: Failed to load skill '{skill_name}': {e}")

    return skills


def parse_skill_file(filepath: Path) -> Optional[Skill]:
    """Parse a single flat skill markdown file."""
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
        name = frontmatter.get('name', filepath.stem)
        description = frontmatter.get('description', '')
        allowed_tools = frontmatter.get('allowed-tools') or frontmatter.get('allowed_tools')

        return Skill(
            name=name,
            description=description,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
            scripts=None,  # Flat-form skills don't have these
            references=None,
            assets=None
        )
    except Exception as e:
        print(f"⚠️  Error parsing skill file {filepath}: {e}")
        return None


def parse_skill_directory(skill_dir: Path) -> Optional[Skill]:
    """Parse a skill from a directory with SKILL.md + scripts/references/assets."""
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        return None

    with open(skill_md, 'r') as f:
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
        name = frontmatter.get('name', skill_dir.name)
        description = frontmatter.get('description', '')
        allowed_tools = frontmatter.get('allowed-tools') or frontmatter.get('allowed_tools')

        # Load resource directories
        scripts = load_resource_dir(skill_dir / "scripts")
        references = load_resource_dir(skill_dir / "references")
        assets = load_resource_dir(skill_dir / "assets")

        return Skill(
            name=name,
            description=description,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
            scripts=scripts,
            references=references,
            assets=assets
        )
    except Exception as e:
        print(f"⚠️  Error parsing skill directory {skill_dir}: {e}")
        return None


def load_resource_dir(resource_path: Path) -> Optional[Dict[str, Path]]:
    """Load a directory of resource files (scripts, references, assets)."""
    if not resource_path.exists():
        return None

    resources = {}
    try:
        for file_path in sorted(resource_path.iterdir()):
            if file_path.is_file():
                resources[file_path.name] = file_path.resolve()
        return resources if resources else None
    except Exception:
        return None
