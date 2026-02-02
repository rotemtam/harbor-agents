# Agent Guidelines for harbor-agents

## Project Overview

This is a Python library that provides a Harbor agent for running Claude Code with custom skills support. The main class `ClaudeCodeWithSkills` extends Harbor's built-in `ClaudeCode` agent.

## Package Manager

This project uses **uv** for package management.

## Common Commands

```bash
# Install dependencies
uv sync --all-extras

# Run unit tests
uv run pytest tests/unit -v

# Run integration tests (requires Harbor setup + API keys)
uv run pytest tests/integration -v

# Run all tests
uv run pytest -v

# Linting
uv run ruff check src tests

# Auto-fix lint issues
uv run ruff check --fix src tests

# Type checking
uv run mypy src

# Build package
uv build
```

## Architecture

### Package Structure

```
src/harbor_agent/
└── skilled_claude/
    ├── __init__.py     # Exports ClaudeCodeWithSkills
    └── agent.py        # Main agent implementation
```

### Key Classes

- **ClaudeCodeWithSkills** (`agent.py`): Extends `harbor.agents.installed.claude_code.ClaudeCode`
  - Accepts `skill_dir` parameter for skills directory path
  - Accepts `skills` parameter for filtering (None=all, ""=none, "a,b"=specific)
  - Copies skills into container's `/workspace/.claude/skills/` during setup

### Test Structure

- `tests/unit/` - Unit tests with mocked `BaseEnvironment`
- `tests/integration/` - Integration tests that run real Harbor evaluations

## Key Behaviors

1. **Skill Detection**: A directory is a valid skill if it contains a `SKILL.md` file at its root
2. **Hidden Directories**: Directories starting with `.` are ignored
3. **Filter Logic**:
   - `skills=None`: Load all valid skills
   - `skills=""`: Load no skills (baseline)
   - `skills="a,b"`: Load only skills named "a" and "b"

## Dependencies

- `harbor` - The Harbor evaluation framework
- `pytest` / `pytest-asyncio` - Testing (dev)
- `ruff` - Linting (dev)
- `mypy` - Type checking (dev)
