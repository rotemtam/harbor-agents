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

### MultiTurnAgent

A composite agent for testing multi-turn conversations. Combines a simulated user (which generates prompts) with an inner agent (which processes them).

```bash
harbor run -p ./my-task \
    --agent-import-path harbor_agent.multi_turn:MultiTurnAgent \
    -m anthropic/claude-sonnet-4-20250514 \
    --ak simulated_user=my_module:MySimulatedUser \
    --ak agent=harbor.agents.installed.claude_code:ClaudeCode \
    --ak 'agent_kwargs={"model_name": "anthropic/claude-sonnet-4-20250514"}' \
    --ak max_turns=10
```

**Options:**

| Option | Description |
|--------|-------------|
| `simulated_user` | Import path to SimulatedUser subclass (`module.path:ClassName`) |
| `agent` | Import path to inner agent (`module.path:ClassName`) |
| `simulated_user_kwargs` | JSON string or dict of kwargs for simulated user |
| `agent_kwargs` | JSON string or dict of kwargs for inner agent |
| `max_turns` | Maximum conversation turns (default: 50) |

**Creating a Simulated User:**

```python
from harbor_agent.multi_turn import SimulatedUser, SimulatedUserDone, ConversationMessage

class MySimulatedUser(SimulatedUser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0

    async def next_message(self, conversation: list[ConversationMessage]) -> str:
        self._turn += 1
        if self._turn > 3:
            raise SimulatedUserDone("Task complete")
        return f"Please do step {self._turn}"
```

**Conversation Flow:**

1. `next_message()` is called with conversation history
2. Returns a prompt string → sent to inner agent
3. Inner agent responds → added to history
4. Repeat until `SimulatedUserDone` is raised or `max_turns` reached

**Output:** Saves `trajectory.json` in ATIF format with all conversation turns.

## Development

```bash
uv sync --all-extras    # Install
uv run pytest -v        # Test
uv run ruff check src   # Lint
uv run mypy src         # Type check
```

## License

Apache 2.0
