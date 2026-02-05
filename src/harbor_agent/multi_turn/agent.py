"""Multi-turn agent for testing with simulated users."""

import importlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent  # type: ignore[import-untyped]
from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]
from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]
from harbor.models.trajectories import Step  # type: ignore[import-untyped]

from harbor_agent.multi_turn.simulated_user import (
    ConversationMessage,
    SimulatedUser,
    SimulatedUserDone,
)

logger = logging.getLogger(__name__)


def _import_class(import_path: str) -> type[Any]:
    """Dynamically import a class from an import path.

    Args:
        import_path: Import path in format "module.path:ClassName"

    Returns:
        The imported class.

    Raises:
        ValueError: If the import path format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the class doesn't exist in the module.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path format: {import_path}. "
            "Expected format: 'module.path:ClassName'"
        )
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls: type[Any] = getattr(module, class_name)
    return cls


def _parse_kwargs(kwargs: dict[str, Any] | str | None) -> dict[str, Any]:
    """Parse kwargs that may be a JSON string or dict.

    Args:
        kwargs: Either a dict, a JSON string, or None.

    Returns:
        Parsed kwargs as a dict.
    """
    if kwargs is None:
        return {}
    if isinstance(kwargs, str):
        return json.loads(kwargs)  # type: ignore[no-any-return]
    return kwargs


class MultiTurnAgent(BaseAgent):  # type: ignore[misc]
    """Agent that runs multi-turn conversations with a simulated user.

    Combines a simulated user (generates prompts) with an inner agent (processes
    them). Each message to the inner agent includes full conversation history,
    so the agent has context from previous turns.

    Args:
        logs_dir: Directory for storing logs.
        simulated_user: Import path to SimulatedUser subclass (module.path:ClassName).
        agent: Import path to the inner agent (module.path:ClassName).
        simulated_user_kwargs: Kwargs for simulated user (dict or JSON string).
        agent_kwargs: Kwargs for inner agent (dict or JSON string).
        max_turns: Maximum conversation turns (default: 50).

    Example:
        harbor run -p ./task \\
            --agent-import-path harbor_agent.multi_turn:MultiTurnAgent \\
            --ak simulated_user=my_module:MySimUser \\
            --ak agent=harbor.agents.installed.claude_code:ClaudeCode \\
            --ak 'agent_kwargs={"model_name": "anthropic/claude-sonnet-4-20250514"}'
    """

    def __init__(
        self,
        logs_dir: Path,
        simulated_user: str,
        agent: str,
        simulated_user_kwargs: dict[str, Any] | str | None = None,
        agent_kwargs: dict[str, Any] | str | None = None,
        max_turns: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, **kwargs)

        self._max_turns = max_turns
        self._conversation: list[ConversationMessage] = []
        self._steps: list[Step] = []
        self._session_id = str(uuid.uuid4())

        # Parse kwargs (handle JSON strings from CLI)
        sim_user_kwargs = _parse_kwargs(simulated_user_kwargs)
        inner_kwargs = _parse_kwargs(agent_kwargs)

        # Import and instantiate simulated user
        sim_user_class = _import_class(simulated_user)
        if not issubclass(sim_user_class, SimulatedUser):
            raise TypeError(
                f"{simulated_user} must be a subclass of SimulatedUser"
            )
        self._simulated_user: SimulatedUser = sim_user_class(**sim_user_kwargs)

        # Import and instantiate inner agent
        inner_agent_class = _import_class(agent)
        # Add logs_dir to inner agent kwargs and ensure it exists
        inner_logs_dir = logs_dir / "inner_agent"
        inner_logs_dir.mkdir(parents=True, exist_ok=True)
        inner_kwargs["logs_dir"] = inner_logs_dir
        self._inner_agent: BaseAgent = inner_agent_class(**inner_kwargs)

    def name(self) -> str:
        return "multi-turn-agent"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await self._inner_agent.setup(environment)
        await self._simulated_user.setup(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the multi-turn conversation loop."""
        self._conversation = []
        self._steps = []
        step_id = 1

        for turn in range(self._max_turns):
            try:
                user_message = await self._simulated_user.next_message(
                    self._conversation
                )
            except SimulatedUserDone:
                break

            self._steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="user",
                    message=user_message,
                    extra={"turn": turn, "simulated": True},
                )
            )
            step_id += 1

            self._conversation.append(
                ConversationMessage(role="user", content=user_message)
            )

            await self._inner_agent.run(user_message, environment, context)

            response = self._read_inner_agent_response()
            if not response:
                logger.warning("No response from inner agent on turn %d", turn)
                response = ""

            self._steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    message=response,
                    extra={"turn": turn},
                )
            )
            step_id += 1

            self._conversation.append(
                ConversationMessage(role="assistant", content=response)
            )

        self._save_trajectory()

    def _read_inner_agent_response(self) -> str:
        """Read the most recent response from the inner agent's log files."""
        inner_logs_dir = self._inner_agent.logs_dir

        def command_number(p: Path) -> int:
            suffix = p.name.removeprefix("command-")
            return int(suffix) if suffix.isdigit() else -1

        command_dirs = sorted(inner_logs_dir.glob("command-*"), key=command_number)

        if command_dirs:
            stdout_file = command_dirs[-1] / "stdout.txt"
            if stdout_file.exists():
                return stdout_file.read_text().strip()

        # Fallback for agents that write to different files
        for name in ["response.txt", "output.txt"]:
            path = inner_logs_dir / name
            if path.exists():
                return path.read_text().strip()

        return ""

    def _save_trajectory(self) -> None:
        """Save the conversation trajectory in ATIF format."""
        inner_agent_name = getattr(self._inner_agent, "name", lambda: "unknown")()

        trajectory_data: dict[str, Any] = {
            "schema_version": "ATIF-v1.5",
            "session_id": self._session_id,
            "agent": {"name": self.name(), "version": self.version()},
            "steps": [step.model_dump() for step in self._steps],
            "final_metrics": {"total_steps": len(self._steps)},
            "extra": {"inner_agent": inner_agent_name, "max_turns": self._max_turns},
        }

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory_data, indent=2)
        )

    @property
    def conversation_history(self) -> list[ConversationMessage]:
        return list(self._conversation)
