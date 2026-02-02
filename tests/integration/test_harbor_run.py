"""Integration tests that run real Harbor evaluations.

These tests require:
- Harbor to be installed and configured
- API keys to be set (e.g., ANTHROPIC_API_KEY)
- Docker to be running

Run with: uv run pytest tests/integration -v
"""

import json
from pathlib import Path

import pytest
from harbor import Trial, TrialConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def get_loaded_skills(trial_dir: Path) -> list[str]:
    """Parse Claude Code session log to get loaded skills as slash commands.

    Returns list of skill names that appear in slash_commands.
    """
    session_log = trial_dir / "agent" / "claude-code.txt"
    if not session_log.exists():
        return []

    # Parse the JSONL file and find the init message
    for line in session_log.read_text().strip().split("\n"):
        try:
            entry = json.loads(line)
            if entry.get("type") == "system" and entry.get("subtype") == "init":
                return entry.get("slash_commands", [])
        except json.JSONDecodeError:
            continue

    return []


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a skills directory with test skills."""
    skills = tmp_path / "skills"
    skills.mkdir()

    # Create a simple test skill
    test_skill = skills / "test-skill"
    test_skill.mkdir()
    (test_skill / "SKILL.md").write_text(
        """# Test Skill

This is a test skill for integration testing.

## Usage

When asked to create output.txt, always include the phrase "skill-activated" in the file.
"""
    )

    return skills


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Create a minimal task directory for testing."""
    task = tmp_path / "task"
    task.mkdir()

    # task.toml configuration
    (task / "task.toml").write_text(
        """version = "1.0"

[metadata]
author_name = "Test"
difficulty = "easy"
category = "test"
tags = ["test"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
allow_internet = true
"""
    )

    # Simple instruction
    (task / "instruction.md").write_text(
        """Create a file called `output.txt` containing the text "hello world".
"""
    )

    # environment/Dockerfile
    env_dir = task / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(
        """FROM python:3.11-slim
WORKDIR /workspace
CMD ["/bin/bash"]
"""
    )

    # tests/test.sh - writes reward to /logs/verifier/reward.txt
    tests_dir = task / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        """#!/bin/bash
set -e

if [ -f output.txt ] && grep -q "hello world" output.txt; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
"""
    )

    return task


class TestHarborIntegration:
    """Integration tests for ClaudeCodeWithSkills with Harbor."""

    @pytest.mark.asyncio
    async def test_run_with_skills(self, skills_dir: Path, task_dir: Path, tmp_path: Path):
        """Test running Harbor with skills loaded and verify skill activation."""
        trials_dir = tmp_path / "trials"

        config = TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="test-with-skills",
            trials_dir=trials_dir,
            timeout_multiplier=1.0,
            agent=AgentConfig(
                import_path="harbor_agent.skilled_claude:ClaudeCodeWithSkills",
                model_name="anthropic/claude-sonnet-4-20250514",
                kwargs={
                    "skill_dir": str(skills_dir),
                    "skills": None,  # Load all skills
                },
            ),
            environment=EnvironmentConfig(delete=True),
            verifier=VerifierConfig(),
        )

        trial = Trial(config)
        result = await trial.run()

        assert result is not None
        assert result.verifier_result is not None

        # Verify skill appears in Claude Code's slash commands
        trial_dir = trials_dir / "test-with-skills"
        slash_commands = get_loaded_skills(trial_dir)
        assert "test-skill" in slash_commands, (
            f"Expected 'test-skill' in slash_commands, got: {slash_commands}"
        )

    @pytest.mark.asyncio
    async def test_run_baseline_no_skills(
        self, skills_dir: Path, task_dir: Path, tmp_path: Path
    ):
        """Test running Harbor without skills (baseline)."""
        trials_dir = tmp_path / "trials"

        config = TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="test-baseline",
            trials_dir=trials_dir,
            timeout_multiplier=1.0,
            agent=AgentConfig(
                import_path="harbor_agent.skilled_claude:ClaudeCodeWithSkills",
                model_name="anthropic/claude-sonnet-4-20250514",
                kwargs={
                    "skill_dir": str(skills_dir),
                    "skills": "",  # Load no skills
                },
            ),
            environment=EnvironmentConfig(delete=True),
            verifier=VerifierConfig(),
        )

        trial = Trial(config)
        result = await trial.run()

        assert result is not None
        assert result.verifier_result is not None

        # Verify NO custom skills were loaded (baseline)
        trial_dir = trials_dir / "test-baseline"
        slash_commands = get_loaded_skills(trial_dir)
        assert "test-skill" not in slash_commands, (
            f"Expected 'test-skill' NOT in slash_commands for baseline, got: {slash_commands}"
        )

    @pytest.mark.asyncio
    async def test_run_with_specific_skill(
        self, skills_dir: Path, task_dir: Path, tmp_path: Path
    ):
        """Test running Harbor with specific skill filter."""
        trials_dir = tmp_path / "trials"

        config = TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="test-specific-skill",
            trials_dir=trials_dir,
            timeout_multiplier=1.0,
            agent=AgentConfig(
                import_path="harbor_agent.skilled_claude:ClaudeCodeWithSkills",
                model_name="anthropic/claude-sonnet-4-20250514",
                kwargs={
                    "skill_dir": str(skills_dir),
                    "skills": "test-skill",  # Only load test-skill
                },
            ),
            environment=EnvironmentConfig(delete=True),
            verifier=VerifierConfig(),
        )

        trial = Trial(config)
        result = await trial.run()

        assert result is not None
        assert result.verifier_result is not None

        # Verify skill appears in Claude Code's slash commands
        trial_dir = trials_dir / "test-specific-skill"
        slash_commands = get_loaded_skills(trial_dir)
        assert "test-skill" in slash_commands, (
            f"Expected 'test-skill' in slash_commands, got: {slash_commands}"
        )
