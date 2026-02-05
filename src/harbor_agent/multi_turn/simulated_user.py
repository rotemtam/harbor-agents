"""Simulated user for multi-turn agent testing."""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]


class ConversationMessage(TypedDict):
    """A message in the conversation history."""

    role: str  # "user" or "assistant"
    content: str


class SimulatedUserDone(Exception):
    """Raised when the simulated user has completed the conversation.

    This exception signals that the multi-turn conversation should terminate.
    The simulated user can optionally include a reason for termination.
    """

    def __init__(self, reason: str = "Conversation complete") -> None:
        """Initialize with an optional reason.

        Args:
            reason: Human-readable reason for ending the conversation.
        """
        self.reason = reason
        super().__init__(reason)


class SimulatedUser(ABC):
    """Abstract base class for simulated users in multi-turn testing.

    A simulated user generates prompts for an agent under test, simulating
    a human user interacting with the agent over multiple turns. This is useful
    for evaluating how well an agent handles multi-step tasks, clarifications,
    and iterative refinement.

    Subclasses must implement the `next_message` method to generate prompts.
    The `setup` method can be overridden for initialization that requires
    environment access.

    Example:
        class ScriptedUser(SimulatedUser):
            def __init__(self, messages: list[str], **kwargs):
                super().__init__(**kwargs)
                self.messages = messages
                self.index = 0

            async def next_message(self, conversation: list[ConversationMessage]) -> str:
                if self.index >= len(self.messages):
                    raise SimulatedUserDone("All messages sent")
                msg = self.messages[self.index]
                self.index += 1
                return msg
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the simulated user.

        Args:
            **kwargs: Configuration options for the simulated user.
        """
        pass

    async def setup(self, environment: BaseEnvironment) -> None:
        """Set up the simulated user with environment access.

        Override this method if your simulated user needs to interact with
        the environment during initialization (e.g., reading task files,
        setting up state).

        Args:
            environment: The Harbor environment for the trial.
        """
        pass

    def set_instruction(self, instruction: str) -> None:
        """Set the task instruction as the goal.

        Called by MultiTurnAgent before the conversation loop starts,
        passing the task instruction. Subclasses can override this to
        use the instruction as their goal or for other purposes.

        Args:
            instruction: The task instruction from the Harbor task.
        """
        pass

    @abstractmethod
    async def next_message(
        self, conversation: list[ConversationMessage]
    ) -> str:
        """Generate the next user message for the conversation.

        This method is called repeatedly during the multi-turn conversation.
        It receives the full conversation history and should return the next
        user message. When the conversation should end, raise SimulatedUserDone.

        Args:
            conversation: List of previous messages, each with 'role' and 'content'.
                The conversation alternates between 'user' and 'assistant' roles.
                Empty list for the first message.

        Returns:
            The next user message to send to the agent.

        Raises:
            SimulatedUserDone: When the conversation should terminate.
        """
        raise NotImplementedError
