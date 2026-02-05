"""Integration tests for MultiTurnAgent with Harbor.

These tests require:
- Harbor to be installed and configured
- API keys to be set (e.g., ANTHROPIC_API_KEY)
- Docker to be running

Run with: uv run pytest tests/integration/test_multi_turn.py -v
"""

import json
from pathlib import Path

import pytest
from harbor import Trial, TrialConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig

from harbor_agent.multi_turn import SimulatedUser, SimulatedUserDone

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class MooSimulatedUser(SimulatedUser):
    """A simulated user that just says 'moo' for testing purposes.

    This cow-themed simulated user sends 3 'moo' messages to Claude,
    then stops the conversation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._moo_count = 0
        self._max_moos = 3

    async def next_message(self, conversation: list) -> str:
        """Send a moo, or stop after 3 moos."""
        if self._moo_count >= self._max_moos:
            raise SimulatedUserDone("Done mooing")
        self._moo_count += 1
        return f"moo #{self._moo_count}"


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Create a minimal task directory for multi-turn testing."""
    task = tmp_path / "task"
    task.mkdir()

    # task.toml configuration
    (task / "task.toml").write_text(
        """version = "1.0"

[metadata]
author_name = "Test"
difficulty = "easy"
category = "test"
tags = ["test", "multi-turn"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 180.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
allow_internet = true
"""
    )

    # Simple instruction (not really used in multi-turn)
    (task / "instruction.md").write_text(
        """This is a multi-turn conversation test.
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

    # tests/test.sh - always passes for multi-turn tests
    tests_dir = task / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        """#!/bin/bash
# Multi-turn tests always pass - we verify via trajectory
echo 1 > /logs/verifier/reward.txt
"""
    )

    return task


class TestMultiTurnIntegration:
    """Integration tests for MultiTurnAgent with Harbor."""

    @pytest.mark.asyncio
    async def test_multi_turn_moo_with_claude(self, task_dir: Path, tmp_path: Path):
        """Test MultiTurnAgent: MooSimulatedUser sends 'moo', Claude responds."""
        trials_dir = tmp_path / "trials"

        config = TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="test-moo-claude",
            trials_dir=trials_dir,
            timeout_multiplier=1.0,
            agent=AgentConfig(
                import_path="harbor_agent.multi_turn:MultiTurnAgent",
                model_name="anthropic/claude-haiku-4-5-20251001",
                kwargs={
                    "simulated_user": (
                        "tests.integration.test_multi_turn:MooSimulatedUser"
                    ),
                    "agent": (
                        "harbor.agents.installed.claude_code:ClaudeCode"
                    ),
                    "agent_kwargs": {
                        "model_name": "anthropic/claude-haiku-4-5-20251001",
                    },
                    "max_turns": 3,
                },
            ),
            environment=EnvironmentConfig(delete=True),
            verifier=VerifierConfig(),
        )

        trial = Trial(config)
        result = await trial.run()

        # Basic result assertions
        assert result is not None
        assert result.verifier_result is not None

        # Check trajectory file exists
        trial_dir = trials_dir / "test-moo-claude"
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        assert trajectory_path.exists(), f"trajectory.json not found at {trajectory_path}"

        # Load and validate trajectory
        trajectory = json.loads(trajectory_path.read_text())

        # Validate trajectory structure
        assert trajectory["schema_version"] == "ATIF-v1.5"
        assert trajectory["agent"]["name"] == "multi-turn-agent"
        assert trajectory["agent"]["version"] == "0.1.0"

        # Validate steps - should have 6 steps (3 user + 3 agent)
        steps = trajectory["steps"]
        assert len(steps) == 6, f"Expected 6 steps, got {len(steps)}"

        # Verify alternating user/agent pattern
        for i, step in enumerate(steps):
            expected_source = "user" if i % 2 == 0 else "agent"
            assert step["source"] == expected_source, (
                f"Step {i} should be {expected_source}, got {step['source']}"
            )

        # Verify user messages are the moos
        user_steps = [s for s in steps if s["source"] == "user"]
        assert user_steps[0]["message"] == "moo #1"
        assert user_steps[1]["message"] == "moo #2"
        assert user_steps[2]["message"] == "moo #3"

        # Verify agent (Claude) responded to each moo
        agent_steps = [s for s in steps if s["source"] == "agent"]
        assert len(agent_steps) == 3, "Claude should have responded 3 times"
        for agent_step in agent_steps:
            # Claude's response should be non-empty
            assert len(agent_step["message"]) > 0, "Claude response should not be empty"

        # Verify final_metrics
        assert trajectory["final_metrics"]["total_steps"] == 6

        # Verify extra metadata
        assert trajectory["extra"]["inner_agent"] == "claude-code"
        assert trajectory["extra"]["max_turns"] == 3

        # Verify simulated flag on user steps
        for user_step in user_steps:
            assert user_step["extra"]["simulated"] is True

        # Verify turn numbers are correct
        for i, step in enumerate(steps):
            expected_turn = i // 2
            assert step["extra"]["turn"] == expected_turn, (
                f"Step {i} should have turn={expected_turn}, got {step['extra']['turn']}"
            )
