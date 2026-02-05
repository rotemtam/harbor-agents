"""Multi-turn agent for testing with simulated users."""

import importlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent  # type: ignore[import-untyped]
from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]
from harbor.models.trajectories import Step  # type: ignore[import-untyped]

from harbor_agent.multi_turn.simulated_user import (
    ConversationMessage,
    SimulatedUser,
    SimulatedUserDone,
)


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

    This composite agent combines a simulated user (which generates prompts)
    with an inner agent (which processes them). It's designed for testing
    how agents handle multi-step tasks and iterative conversations.

    The conversation loop:
    1. Ask the simulated user for the next message
    2. Send the message to the inner agent
    3. Get the agent's response
    4. Repeat until SimulatedUserDone is raised

    Args:
        logs_dir: Directory for storing logs.
        simulated_user_import_path: Import path to SimulatedUser subclass,
            format: "module.path:ClassName"
        inner_agent_import_path: Import path to the agent under test,
            format: "module.path:ClassName"
        simulated_user_kwargs: Kwargs to pass to simulated user constructor.
            Can be a dict or JSON string.
        inner_agent_kwargs: Kwargs to pass to inner agent constructor.
            Can be a dict or JSON string.
        max_turns: Maximum number of conversation turns (default: 50).
        **kwargs: Additional arguments passed to BaseAgent.

    Example CLI usage:
        harbor run -p ./task \\
            --agent-import-path harbor_agent.multi_turn:MultiTurnAgent \\
            -m anthropic/claude-sonnet-4-20250514 \\
            --ak simulated_user_import_path=my_module:MySimUser \\
            --ak inner_agent_import_path=harbor_agent.skilled_claude:ClaudeCodeWithSkills \\
            --ak 'inner_agent_kwargs={"skill_dir": "./skills"}'
    """

    def __init__(
        self,
        logs_dir: Path,
        simulated_user_import_path: str,
        inner_agent_import_path: str,
        simulated_user_kwargs: dict[str, Any] | str | None = None,
        inner_agent_kwargs: dict[str, Any] | str | None = None,
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
        inner_kwargs = _parse_kwargs(inner_agent_kwargs)

        # Import and instantiate simulated user
        sim_user_class = _import_class(simulated_user_import_path)
        if not issubclass(sim_user_class, SimulatedUser):
            raise TypeError(
                f"{simulated_user_import_path} must be a subclass of SimulatedUser"
            )
        self._simulated_user: SimulatedUser = sim_user_class(**sim_user_kwargs)

        # Import and instantiate inner agent
        inner_agent_class = _import_class(inner_agent_import_path)
        # Add logs_dir to inner agent kwargs
        inner_kwargs["logs_dir"] = logs_dir / "inner_agent"
        self._inner_agent: BaseAgent = inner_agent_class(**inner_kwargs)

    def name(self) -> str:
        """Return the agent name."""
        return "multi-turn-agent"

    def version(self) -> str:
        """Return the agent version."""
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Set up both the simulated user and inner agent.

        Args:
            environment: The Harbor environment for the trial.
        """
        # Set up inner agent first
        await self._inner_agent.setup(environment)

        # Set up simulated user (may need environment access)
        await self._simulated_user.setup(environment)

        # Store environment for the conversation loop
        self._environment = environment

    async def run(self, instruction: str) -> str:
        """Run the multi-turn conversation loop.

        Args:
            instruction: The task instruction (passed to simulated user context).

        Returns:
            The final response from the conversation.
        """
        # Initialize conversation and trajectory tracking
        self._conversation = []
        self._steps = []
        step_id = 1
        last_response = ""

        for turn in range(self._max_turns):
            try:
                # Get next message from simulated user
                user_message = await self._simulated_user.next_message(
                    self._conversation
                )
            except SimulatedUserDone:
                # Conversation complete
                break

            # Log simulated user turn as ATIF Step
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

            # Add user message to history
            self._conversation.append(
                ConversationMessage(role="user", content=user_message)
            )

            # Send to inner agent and get response
            response = await self._inner_agent.run(user_message)
            last_response = response

            # Log agent response as ATIF Step
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

            # Add assistant response to history
            self._conversation.append(
                ConversationMessage(role="assistant", content=response)
            )

        # Save the trajectory
        self._save_trajectory()

        return last_response

    def _save_trajectory(self) -> None:
        """Save the conversation trajectory in ATIF format."""
        # Get inner agent name if available
        inner_agent_name = getattr(self._inner_agent, "name", lambda: "unknown")()

        # Build trajectory data
        trajectory_data: dict[str, Any] = {
            "schema_version": "ATIF-v1.5",
            "session_id": self._session_id,
            "agent": {
                "name": self.name(),
                "version": self.version(),
            },
            "steps": [step.model_dump() for step in self._steps],
            "final_metrics": {
                "total_steps": len(self._steps),
            },
            "extra": {
                "inner_agent": inner_agent_name,
                "max_turns": self._max_turns,
            },
        }

        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory_data, indent=2))

    @property
    def conversation_history(self) -> list[ConversationMessage]:
        """Get the conversation history.

        Returns:
            List of conversation messages.
        """
        return list(self._conversation)
