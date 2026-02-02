# Harbor Agents

A collection of custom agents for [Harbor](https://harborframework.com/) - the AI agent evaluation framework.

## Installation

```bash
pip install harbor-agents
```

## Available Agents

### ClaudeCodeWithSkills

Extends Claude Code with custom skills support. Load pre-configured skills into the container for evaluations.

```bash
# Load all skills from a directory
harbor run -p ./my-task \
    --agent-import-path harbor_agent.skilled_claude:ClaudeCodeWithSkills \
    -m anthropic/claude-sonnet-4-20250514 \
    --ak skill_dir=./skills

# Load specific skills only
harbor run -p ./my-task \
    --agent-import-path harbor_agent.skilled_claude:ClaudeCodeWithSkills \
    -m anthropic/claude-sonnet-4-20250514 \
    --ak skill_dir=./skills \
    --ak skills=my-skill,another-skill

# Baseline (no skills)
harbor run -p ./my-task \
    --agent-import-path harbor_agent.skilled_claude:ClaudeCodeWithSkills \
    -m anthropic/claude-sonnet-4-20250514 \
    --ak skill_dir=./skills \
    --ak 'skills='
```

**Options:**

| Option | Description |
|--------|-------------|
| `skill_dir` | Path to directory containing skill folders |
| `skills` | Filter: omit for all, `skill-a,skill-b` for specific, empty string for none |

**Skill Directory Structure:**

```
skills/
├── my-skill/
│   ├── SKILL.md          # Required
│   └── references/       # Optional
└── another-skill/
    └── SKILL.md
```

## Development

```bash
uv sync --all-extras    # Install
uv run pytest -v        # Test
uv run ruff check src   # Lint
uv run mypy src         # Type check
```

## License

Apache 2.0
