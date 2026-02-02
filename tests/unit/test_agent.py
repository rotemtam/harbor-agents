"""Unit tests for ClaudeCodeWithSkills agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harbor_agent.skilled_claude import ClaudeCodeWithSkills


class TestClaudeCodeWithSkills:
    """Tests for ClaudeCodeWithSkills agent."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock BaseEnvironment for testing."""
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", return_code=0))
        env.upload_dir = AsyncMock()
        return env

    def test_name(self, tmp_path):
        """Test agent name."""
        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs")
        assert agent.name() == "claude-code-with-skills"

    def test_should_load_skill_with_none_filter(self, tmp_path):
        """skills=None should load all skills."""
        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skills=None)
        assert agent._should_load_skill("any-skill") is True
        assert agent._should_load_skill("another-skill") is True

    def test_should_load_skill_with_empty_filter(self, tmp_path):
        """skills='' should load no skills."""
        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skills="")
        assert agent._should_load_skill("any-skill") is False
        assert agent._should_load_skill("another-skill") is False

    def test_should_load_skill_with_specific_filter(self, tmp_path):
        """skills='a,b' should only load specified skills."""
        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skills="skill-a,skill-b")
        assert agent._should_load_skill("skill-a") is True
        assert agent._should_load_skill("skill-b") is True
        assert agent._should_load_skill("skill-c") is False

    @pytest.mark.asyncio
    async def test_setup_creates_skills_directory(self, mock_environment, tmp_path):
        """Setup should create skills directory in container."""
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=tmp_path / "skills")

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.exec.assert_any_call(command="mkdir -p /workspace/.claude/skills")

    @pytest.mark.asyncio
    async def test_setup_uploads_skill_directory(self, mock_environment, tmp_path):
        """Setup should upload skill directory to container."""
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=tmp_path / "skills")

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_called_once_with(
            source_dir=skill_dir,
            target_dir="/workspace/.claude/skills/my-skill",
        )

    @pytest.mark.asyncio
    async def test_skills_filter_loads_only_specified(self, mock_environment, tmp_path):
        """Filter should only load specified skills."""
        for name in ["skill-a", "skill-b", "skill-c"]:
            skill_path = tmp_path / "skills" / name
            skill_path.mkdir(parents=True)
            (skill_path / "SKILL.md").write_text(f"# {name}")

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=tmp_path / "skills",
            skills="skill-a,skill-c",
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        calls = mock_environment.upload_dir.call_args_list
        assert len(calls) == 2
        uploaded_skills = {call.kwargs["target_dir"].split("/")[-1] for call in calls}
        assert uploaded_skills == {"skill-a", "skill-c"}

    @pytest.mark.asyncio
    async def test_skills_empty_string_loads_none(self, mock_environment, tmp_path):
        """Empty skills string should load no skills."""
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill")

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=tmp_path / "skills",
            skills="",
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_directories_without_skill_md(self, mock_environment, tmp_path):
        """Directories without SKILL.md should be skipped."""
        valid_skill = tmp_path / "skills" / "valid"
        valid_skill.mkdir(parents=True)
        (valid_skill / "SKILL.md").write_text("# Valid")

        invalid_skill = tmp_path / "skills" / "invalid"
        invalid_skill.mkdir(parents=True)
        (invalid_skill / "README.md").write_text("# Not a skill")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=tmp_path / "skills")

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_called_once()
        assert "valid" in mock_environment.upload_dir.call_args.kwargs["target_dir"]

    @pytest.mark.asyncio
    async def test_skill_dir_does_not_exist(self, mock_environment, tmp_path):
        """Non-existent skill_dir should not error, just skip."""
        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=tmp_path / "nonexistent",
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_skill_dir_is_file_not_directory(self, mock_environment, tmp_path):
        """skill_dir pointing to a file should be skipped."""
        skill_file = tmp_path / "skills"
        skill_file.write_text("I'm a file")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=skill_file)

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_skill_directory(self, mock_environment, tmp_path):
        """Empty skill directory should result in no uploads."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=skill_dir)

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_skill_filter_with_whitespace(self, mock_environment, tmp_path):
        """Skill filter with whitespace should be trimmed."""
        for name in ["skill-a", "skill-b"]:
            skill_path = tmp_path / "skills" / name
            skill_path.mkdir(parents=True)
            (skill_path / "SKILL.md").write_text(f"# {name}")

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=tmp_path / "skills",
            skills="  skill-a  ,  skill-b  ",
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        assert mock_environment.upload_dir.call_count == 2

    @pytest.mark.asyncio
    async def test_skill_filter_with_nonexistent_skill(self, mock_environment, tmp_path):
        """Filter with non-existent skill should only load existing ones."""
        skill_path = tmp_path / "skills" / "real-skill"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("# Real")

        agent = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs",
            skill_dir=tmp_path / "skills",
            skills="real-skill,fake-skill",
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_called_once()
        assert "real-skill" in mock_environment.upload_dir.call_args.kwargs["target_dir"]

    @pytest.mark.asyncio
    async def test_nested_skill_md_not_detected(self, mock_environment, tmp_path):
        """SKILL.md in nested directory should not make parent a skill."""
        skill_path = tmp_path / "skills" / "not-a-skill" / "nested"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("# Nested")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=tmp_path / "skills")

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_skill_with_nested_structure(self, mock_environment, tmp_path):
        """Skills with nested directories should upload correctly."""
        skill_path = tmp_path / "skills" / "complex-skill"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("# Complex")

        refs_path = skill_path / "references"
        refs_path.mkdir()
        (refs_path / "guide.md").write_text("# Guide")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=tmp_path / "skills")

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_called_once_with(
            source_dir=skill_path,
            target_dir="/workspace/.claude/skills/complex-skill",
        )

    @pytest.mark.asyncio
    async def test_hidden_directories_ignored(self, mock_environment, tmp_path):
        """Hidden directories should not be treated as skills."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()

        # Hidden directory (should be ignored)
        hidden = skill_dir / ".hidden"
        hidden.mkdir()
        (hidden / "SKILL.md").write_text("# Hidden")

        # Real skill
        real = skill_dir / "real-skill"
        real.mkdir()
        (real / "SKILL.md").write_text("# Real")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=skill_dir)

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_called_once()
        assert "real-skill" in mock_environment.upload_dir.call_args.kwargs["target_dir"]

    @pytest.mark.asyncio
    async def test_skill_none_vs_empty_string(self, mock_environment, tmp_path):
        """skills=None loads all, skills='' loads none."""
        skill_path = tmp_path / "skills" / "test-skill"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("# Test")

        # None = load all
        agent_all = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs1",
            skill_dir=tmp_path / "skills",
            skills=None,
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent_all.setup(mock_environment)

        assert mock_environment.upload_dir.call_count == 1

        mock_environment.reset_mock()

        # Empty string = load none (baseline)
        agent_none = ClaudeCodeWithSkills(
            logs_dir=tmp_path / "logs2",
            skill_dir=tmp_path / "skills",
            skills="",
        )

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent_none.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_skill_dir_provided(self, mock_environment, tmp_path):
        """skill_dir=None should work (baseline without skills)."""
        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=None)

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_not_called()

    @pytest.mark.asyncio
    async def test_special_characters_in_skill_name(self, mock_environment, tmp_path):
        """Skills with special characters should work."""
        skill_path = tmp_path / "skills" / "my_skill-v2.0"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("# Special")

        agent = ClaudeCodeWithSkills(logs_dir=tmp_path / "logs", skill_dir=tmp_path / "skills")

        with patch.object(ClaudeCodeWithSkills.__bases__[0], "setup", new_callable=AsyncMock):
            await agent.setup(mock_environment)

        mock_environment.upload_dir.assert_called_once()
        assert "my_skill-v2.0" in mock_environment.upload_dir.call_args.kwargs["target_dir"]
