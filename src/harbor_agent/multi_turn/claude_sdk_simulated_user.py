"""Claude Agent SDK powered simulated user."""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

from harbor_agent.multi_turn.simulated_user import (
    ConversationMessage,
    SimulatedUser,
    SimulatedUserDone,
)

QueryFn = Callable[..., AsyncIterator[Any]]


class ClaudeSdkSimulatedUser(SimulatedUser):
    """Simulated user backed by the Claude Agent SDK.

    Uses the Claude Agent SDK `query()` function to generate the next user message
    based on the conversation history and a user goal. The model is instructed to
    return a special done token when the conversation should end.
    """

    def __init__(
        self,
        goal: str = "",
        system_prompt: str | None = None,
        done_token: str = "<<DONE>>",
        max_turns: int = 1,
        agent_options: dict[str, Any] | Any | None = None,
        query_fn: QueryFn | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the simulated user.

        Args:
            goal: User goal for the simulated conversation. If empty, the task
                instruction will be used as the goal via set_instruction().
            system_prompt: Optional system prompt override for the SDK.
            done_token: Token that signals the simulated user is done.
            max_turns: Max turns for the Claude Agent SDK internal loop.
            agent_options: Options for ClaudeAgentOptions or a prebuilt options object.
            query_fn: Optional injected query function (useful for tests).
            **kwargs: Additional configuration.
        """
        super().__init__(**kwargs)
        if not done_token.strip():
            raise ValueError("done_token must be a non-empty string")

        self._goal = goal
        self._done_token = done_token
        self._max_turns = max_turns
        self._agent_options_input = agent_options
        self._query_fn = query_fn
        self._assistant_message_type: type[Any] | None = None
        self._text_block_type: type[Any] | None = None
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._agent_options: Any | None = None

    def set_instruction(self, instruction: str) -> None:
        """Set the task instruction as the goal if no goal was provided.

        Args:
            instruction: The task instruction from the Harbor task.
        """
        if not self._goal:
            self._goal = instruction

    async def next_message(self, conversation: list[ConversationMessage]) -> str:
        if not self._goal:
            raise ValueError(
                "goal must be set either via constructor or set_instruction()"
            )
        self._ensure_sdk_loaded()
        prompt = self._build_prompt(conversation)
        response_text = await self._run_query(prompt)

        if not response_text:
            raise SimulatedUserDone("No assistant text returned")

        if self._is_done_response(response_text):
            raise SimulatedUserDone("Claude Agent SDK signaled completion")

        return response_text

    def _ensure_sdk_loaded(self) -> None:
        if self._query_fn is None:
            try:
                from claude_agent_sdk import (  # type: ignore[import-untyped]
                    AssistantMessage,
                    ClaudeAgentOptions,
                    TextBlock,
                    query,
                )
            except Exception as exc:
                raise ImportError(
                    "claude-agent-sdk is required to use ClaudeSdkSimulatedUser. "
                    "Install with `pip install claude-agent-sdk`."
                ) from exc

            self._query_fn = query
            self._assistant_message_type = AssistantMessage
            self._text_block_type = TextBlock
            self._agent_options = self._build_agent_options(ClaudeAgentOptions)
        elif self._agent_options is None:
            # When a query function is injected (tests), keep options as plain data.
            self._agent_options = self._build_fallback_options()

    def _build_agent_options(self, options_cls: type[Any]) -> Any:
        if self._agent_options_input is None:
            return options_cls(**self._default_options_dict())
        if isinstance(self._agent_options_input, dict):
            options = dict(self._agent_options_input)
            options.setdefault("system_prompt", self._system_prompt)
            options.setdefault("max_turns", self._max_turns)
            return options_cls(**options)
        return self._agent_options_input

    def _build_fallback_options(self) -> Any:
        if self._agent_options_input is None:
            return self._default_options_dict()
        if isinstance(self._agent_options_input, dict):
            options = dict(self._agent_options_input)
            options.setdefault("system_prompt", self._system_prompt)
            options.setdefault("max_turns", self._max_turns)
            return options
        return self._agent_options_input

    def _default_options_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self._system_prompt,
            "max_turns": self._max_turns,
        }

    def _default_system_prompt(self) -> str:
        return (
            "You are simulating a human user in a multi-turn conversation with an "
            "assistant. Provide only the next user message, based on the user's goal "
            "and the conversation so far. Do not include analysis or meta-commentary. "
            f"If the user's goal is complete, respond with the exact token "
            f"{self._done_token} and nothing else."
        )

    def _build_prompt(self, conversation: list[ConversationMessage]) -> str:
        transcript = self._format_conversation(conversation)
        return (
            f"User goal:\n{self._goal}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            "Next user message:"
        )

    def _format_conversation(self, conversation: list[ConversationMessage]) -> str:
        if not conversation:
            return "(no prior messages)"
        lines: list[str] = []
        for message in conversation:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _run_query(self, prompt: str) -> str:
        assert self._query_fn is not None
        response_parts: list[str] = []
        async for message in self._query_fn(prompt=prompt, options=self._agent_options):
            if not self._is_assistant_message(message):
                continue
            for block in self._get_content_blocks(message):
                if self._is_text_block(block):
                    text = self._get_text(block)
                    if text:
                        response_parts.append(text)
        return "".join(response_parts).strip()

    def _is_assistant_message(self, message: Any) -> bool:
        if self._assistant_message_type is not None:
            return isinstance(message, self._assistant_message_type)
        msg_type = self._get_attr(message, "type")
        if msg_type is not None:
            return msg_type == "assistant"
        return hasattr(message, "content")

    def _get_content_blocks(self, message: Any) -> list[Any]:
        blocks = self._get_attr(message, "content")
        if blocks is None:
            return []
        return list(blocks)

    def _is_text_block(self, block: Any) -> bool:
        if self._text_block_type is not None:
            return isinstance(block, self._text_block_type)
        block_type = self._get_attr(block, "type")
        if block_type is not None:
            return block_type == "text"
        return hasattr(block, "text")

    def _get_text(self, block: Any) -> str:
        text = self._get_attr(block, "text")
        return text if isinstance(text, str) else ""

    def _get_attr(self, obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _is_done_response(self, response_text: str) -> bool:
        normalized = response_text.strip()
        if normalized == self._done_token:
            return True
        if normalized.startswith(f"{self._done_token} "):
            return True
        if normalized.startswith(f"{self._done_token}\n"):
            return True
        return False
