"""Claude Code agent with skills support."""

from pathlib import Path
from typing import Any

from harbor.agents.installed.claude_code import ClaudeCode  # type: ignore[import-untyped]
from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]


class ClaudeCodeWithSkills(ClaudeCode):  # type: ignore[misc]
    """Claude Code agent that loads custom skills into the container.

    This agent extends the standard ClaudeCode agent by copying skill directories
    into the container's `/workspace/.claude/skills/` directory during setup.

    Args:
        logs_dir: Directory for storing logs.
        skill_dir: Path to directory containing skill folders. Each skill folder
            must contain a SKILL.md file to be recognized.
        skills: Comma-separated list of skill names to load. Use None to load all
            skills, or an empty string to load no skills (baseline mode).
        **kwargs: Additional arguments passed to ClaudeCode.
    """

    def __init__(
        self,
        logs_dir: Path,
        skill_dir: Path | str | None = None,
        skills: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, **kwargs)
        self._skill_dir = Path(skill_dir) if skill_dir else None

        # Parse skills filter: None means all, empty string means none
        if skills is None:
            self._skills_filter: set[str] | None = None  # Load all
        elif skills == "":
            self._skills_filter = set()  # Load none (baseline)
        else:
            # Strip whitespace from each skill name
            self._skills_filter = {s.strip() for s in skills.split(",") if s.strip()}

    def name(self) -> str:
        """Return the agent name."""
        return "claude-code-with-skills"

    def _should_load_skill(self, skill_name: str) -> bool:
        """Check if a skill should be loaded based on the filter.

        Args:
            skill_name: Name of the skill to check.

        Returns:
            True if the skill should be loaded, False otherwise.
        """
        if self._skills_filter is None:
            return True  # Load all
        return skill_name in self._skills_filter

    async def setup(self, environment: BaseEnvironment) -> None:
        """Set up the agent in the environment.

        Calls parent setup and then copies skill directories into the container.

        Args:
            environment: The environment to set up in.
        """
        await super().setup(environment)

        if (
            not self._skill_dir
            or not self._skill_dir.exists()
            or not self._skill_dir.is_dir()
        ):
            return

        skills_to_load: list[Path] = []
        for skill_path in self._skill_dir.iterdir():
            # Skip hidden directories, non-directories, and dirs without SKILL.md
            if (
                skill_path.is_dir()
                and not skill_path.name.startswith(".")
                and (skill_path / "SKILL.md").exists()
                and self._should_load_skill(skill_path.name)
            ):
                skills_to_load.append(skill_path)

        if not skills_to_load:
            return

        # Create parent directory only if we have skills to load
        await environment.exec(command="mkdir -p /workspace/.claude/skills")

        for skill_path in skills_to_load:
            await environment.upload_dir(
                source_dir=skill_path,
                target_dir=f"/workspace/.claude/skills/{skill_path.name}",
            )
