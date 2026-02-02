"""Integration tests that run real Harbor evaluations.

These tests require:
- Harbor to be installed and configured
- API keys to be set (e.g., ANTHROPIC_API_KEY)
- Docker to be running

Run with: uv run pytest tests/integration -v
"""

from pathlib import Path

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


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

Use this skill when the user asks about testing.
"""
    )

    return skills


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Create a minimal task directory for testing."""
    task = tmp_path / "task"
    task.mkdir()

    # Minimal Dockerfile
    (task / "Dockerfile").write_text(
        """FROM python:3.11-slim
WORKDIR /workspace
"""
    )

    # Simple instruction
    (task / "instruction.md").write_text(
        """Create a file called `output.txt` containing the text "hello world".
"""
    )

    # Test script
    (task / "test.sh").write_text(
        """#!/bin/bash
if [ -f output.txt ] && grep -q "hello world" output.txt; then
    echo "PASS"
    exit 0
else
    echo "FAIL"
    exit 1
fi
"""
    )

    return task


class TestHarborIntegration:
    """Integration tests for ClaudeCodeWithSkills with Harbor."""

    @pytest.mark.skip(reason="Requires Harbor setup and API keys")
    @pytest.mark.asyncio
    async def test_run_with_skills(self, skills_dir: Path, task_dir: Path, tmp_path: Path):
        """Test running Harbor with skills loaded."""
        from harbor.run import run_task

        from harbor_agent.skilled_claude import ClaudeCodeWithSkills

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=skills_dir,
            skills=None,  # Load all skills
        )

        result = await run_task(
            agent=agent,
            task_dir=task_dir,
        )

        assert result is not None

    @pytest.mark.skip(reason="Requires Harbor setup and API keys")
    @pytest.mark.asyncio
    async def test_run_baseline_no_skills(
        self, skills_dir: Path, task_dir: Path, tmp_path: Path
    ):
        """Test running Harbor without skills (baseline)."""
        from harbor.run import run_task

        from harbor_agent.skilled_claude import ClaudeCodeWithSkills

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=skills_dir,
            skills="",  # Load no skills
        )

        result = await run_task(
            agent=agent,
            task_dir=task_dir,
        )

        assert result is not None

    @pytest.mark.skip(reason="Requires Harbor setup and API keys")
    @pytest.mark.asyncio
    async def test_run_with_specific_skill(
        self, skills_dir: Path, task_dir: Path, tmp_path: Path
    ):
        """Test running Harbor with specific skill filter."""
        from harbor.run import run_task

        from harbor_agent.skilled_claude import ClaudeCodeWithSkills

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=skills_dir,
            skills="test-skill",  # Only load test-skill
        )

        result = await run_task(
            agent=agent,
            task_dir=task_dir,
        )

        assert result is not None
