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
from harbor.agents.base import BaseAgent  # type: ignore[import-untyped]
from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig

from harbor_agent.multi_turn import SimulatedUser, SimulatedUserDone

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class MooAgent(BaseAgent):  # type: ignore[misc]
    """A simple agent that just says 'moo' for testing purposes."""

    def __init__(self, logs_dir: Path, **kwargs):
        super().__init__(logs_dir=logs_dir, **kwargs)
        self._moo_count = 0

    def name(self) -> str:
        return "moo-agent"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """No setup needed for the moo agent."""
        pass

    async def run(self, message: str) -> str:
        """Always respond with 'moo'."""
        self._moo_count += 1
        return f"moo #{self._moo_count}"


class ThreeTurnSimUser(SimulatedUser):
    """A simulated user that asks exactly 3 questions then stops."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0
        self._questions = [
            "What sound does a cow make?",
            "Can you say it again?",
            "One more time please!",
        ]

    async def next_message(self, conversation: list) -> str:
        if self._turn >= len(self._questions):
            raise SimulatedUserDone("Asked all 3 questions")
        question = self._questions[self._turn]
        self._turn += 1
        return question


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
timeout_sec = 120.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
allow_internet = false
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
    async def test_multi_turn_with_moo_agent(self, task_dir: Path, tmp_path: Path):
        """Test MultiTurnAgent with MooAgent for 3 turns."""
        trials_dir = tmp_path / "trials"

        config = TrialConfig(
            task=TaskConfig(path=task_dir),
            trial_name="test-multi-turn-moo",
            trials_dir=trials_dir,
            timeout_multiplier=1.0,
            agent=AgentConfig(
                import_path="harbor_agent.multi_turn:MultiTurnAgent",
                model_name="anthropic/claude-haiku-4-5-20251001",
                kwargs={
                    "simulated_user_import_path": (
                        "tests.integration.test_multi_turn:ThreeTurnSimUser"
                    ),
                    "inner_agent_import_path": (
                        "tests.integration.test_multi_turn:MooAgent"
                    ),
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
        trial_dir = trials_dir / "test-multi-turn-moo"
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

        # Verify user messages are the questions
        user_steps = [s for s in steps if s["source"] == "user"]
        assert user_steps[0]["message"] == "What sound does a cow make?"
        assert user_steps[1]["message"] == "Can you say it again?"
        assert user_steps[2]["message"] == "One more time please!"

        # Verify agent responses contain "moo"
        agent_steps = [s for s in steps if s["source"] == "agent"]
        for agent_step in agent_steps:
            assert "moo" in agent_step["message"].lower(), (
                f"Expected 'moo' in response, got: {agent_step['message']}"
            )

        # Verify final_metrics
        assert trajectory["final_metrics"]["total_steps"] == 6

        # Verify extra metadata
        assert trajectory["extra"]["inner_agent"] == "moo-agent"
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
