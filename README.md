# Harbor Agent: Claude Code with Skills

A Harbor agent that extends Claude Code with custom skills support. This library allows you to run Claude Code evaluations with pre-configured skills loaded into the container.

## Installation

```bash
uv add harbor-agent-skilled-claude
```

Or with pip:

```bash
pip install harbor-agent-skilled-claude
```

## Usage

### Basic Usage

```python
from harbor_agent.skilled_claude import ClaudeCodeWithSkills
from pathlib import Path

# Create agent with skills directory
agent = ClaudeCodeWithSkills(
    logs_dir=Path("./logs"),
    skill_dir=Path("./skills"),  # Directory containing skill folders
)
```

### Skill Filtering

```python
# Load all skills (default)
agent = ClaudeCodeWithSkills(
    logs_dir=Path("./logs"),
    skill_dir=Path("./skills"),
    skills=None,  # Loads all valid skills
)

# Load specific skills only
agent = ClaudeCodeWithSkills(
    logs_dir=Path("./logs"),
    skill_dir=Path("./skills"),
    skills="skill-a,skill-b",  # Only loads skill-a and skill-b
)

# Baseline mode (no skills)
agent = ClaudeCodeWithSkills(
    logs_dir=Path("./logs"),
    skill_dir=Path("./skills"),
    skills="",  # Empty string = load no skills
)
```

### With Harbor CLI

Use this agent with Harbor's run command by specifying it as the agent:

```bash
harbor run --agent harbor_agent.skilled_claude:ClaudeCodeWithSkills --task ./my-task
```

## Skill Directory Structure

Skills should be organized in a directory structure like:

```
skills/
├── my-skill/
│   ├── SKILL.md          # Required - skill definition
│   └── references/       # Optional - additional resources
│       └── guide.md
├── another-skill/
│   └── SKILL.md
```

Each skill directory **must** contain a `SKILL.md` file to be recognized as a valid skill.

## Development

This project uses `uv` for package management.

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd harbor-agents

# Install dependencies
uv sync --all-extras
```

### Running Tests

```bash
# Unit tests
uv run pytest tests/unit -v

# Integration tests (requires Harbor + API keys)
uv run pytest tests/integration -v

# All tests
uv run pytest -v
```

### Linting and Type Checking

```bash
# Linting
uv run ruff check src tests

# Type checking
uv run mypy src
```

### Building

```bash
uv build
```

## License

MIT
