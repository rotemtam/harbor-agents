"""Unit tests for MultiTurnAgent and SimulatedUser."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harbor_agent.multi_turn import (
    ConversationMessage,
    MultiTurnAgent,
    SimulatedUser,
    SimulatedUserDone,
)


class MockHarborInnerAgent:
    """A mock inner agent that simulates Harbor's file-based output pattern.

    Writes responses to command-N/stdout.txt files like ClaudeCode does.
    """

    def __init__(
        self,
        logs_dir: Path,
        response_fn: Any = None,
        agent_name: str = "mock-inner-agent",
        **kwargs: Any,
    ):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._response_fn = response_fn or (lambda msg: f"Response to: {msg}")
        self._agent_name = agent_name
        self._command_count = 0
        self.kwargs = kwargs
        self.setup_called = False
        self.messages_received: list[str] = []

    def name(self) -> str:
        return self._agent_name

    async def setup(self, env: Any) -> None:
        self.setup_called = True

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        """Simulate Harbor agent behavior: write response to files."""
        self.messages_received.append(instruction)
        response = self._response_fn(instruction)

        command_dir = self.logs_dir / f"command-{self._command_count}"
        command_dir.mkdir(parents=True, exist_ok=True)
        (command_dir / "stdout.txt").write_text(response)
        self._command_count += 1


class TestSimulatedUserDone:
    """Tests for SimulatedUserDone exception."""

    def test_default_reason(self):
        """Exception should have default reason."""
        exc = SimulatedUserDone()
        assert exc.reason == "Conversation complete"
        assert str(exc) == "Conversation complete"

    def test_custom_reason(self):
        """Exception should accept custom reason."""
        exc = SimulatedUserDone("Task completed successfully")
        assert exc.reason == "Task completed successfully"
        assert str(exc) == "Task completed successfully"


class TestSimulatedUser:
    """Tests for SimulatedUser ABC."""

    def test_is_abstract(self):
        """SimulatedUser should be abstract."""
        with pytest.raises(TypeError, match="abstract"):
            SimulatedUser()  # type: ignore[abstract]

    def test_subclass_must_implement_next_message(self):
        """Subclass without next_message should fail."""

        class IncompleteUser(SimulatedUser):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteUser()  # type: ignore[abstract]

    def test_subclass_can_be_instantiated(self):
        """Complete subclass should be instantiatable."""

        class CompleteUser(SimulatedUser):
            async def next_message(
                self, conversation: list[ConversationMessage]
            ) -> str:
                return "test"

        user = CompleteUser()
        assert user is not None

    @pytest.mark.asyncio
    async def test_setup_is_optional(self):
        """Setup method should be optional (has default implementation)."""

        class MinimalUser(SimulatedUser):
            async def next_message(
                self, conversation: list[ConversationMessage]
            ) -> str:
                return "test"

        user = MinimalUser()
        env = MagicMock()
        # Should not raise
        await user.setup(env)


class ScriptedSimUser(SimulatedUser):
    """A simple scripted simulated user for testing."""

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


class TestScriptedSimUser:
    """Tests for ScriptedSimUser helper class."""

    @pytest.mark.asyncio
    async def test_returns_messages_in_order(self):
        """Should return messages in order."""
        user = ScriptedSimUser(["first", "second", "third"])

        assert await user.next_message([]) == "first"
        assert await user.next_message([]) == "second"
        assert await user.next_message([]) == "third"

    @pytest.mark.asyncio
    async def test_raises_done_when_exhausted(self):
        """Should raise SimulatedUserDone when messages exhausted."""
        user = ScriptedSimUser(["only one"])

        await user.next_message([])
        with pytest.raises(SimulatedUserDone):
            await user.next_message([])

    @pytest.mark.asyncio
    async def test_empty_messages_raises_immediately(self):
        """Empty message list should raise immediately."""
        user = ScriptedSimUser([])

        with pytest.raises(SimulatedUserDone):
            await user.next_message([])


class TestMultiTurnAgent:
    """Tests for MultiTurnAgent."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock BaseEnvironment for testing."""
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", return_code=0))
        env.upload_dir = AsyncMock()
        return env

    @pytest.fixture
    def mock_inner_agent_class(self):
        """Create a mock inner agent class."""
        return MockHarborInnerAgent

    def test_name(self, tmp_path, mock_inner_agent_class):
        """Test agent name."""
        with patch.dict(
            "sys.modules",
            {"test_module": MagicMock(TestSimUser=ScriptedSimUser)},
        ):
            with patch(
                "harbor_agent.multi_turn.agent._import_class"
            ) as mock_import:
                mock_import.side_effect = [ScriptedSimUser, mock_inner_agent_class]

                agent = MultiTurnAgent(
                    logs_dir=tmp_path / "logs",
                    simulated_user="test_module:TestSimUser",
                    agent="test_module:MockAgent",
                    simulated_user_kwargs={"messages": ["hello"]},
                )

                assert agent.name() == "multi-turn-agent"

    def test_invalid_import_path_format(self, tmp_path):
        """Should raise ValueError for invalid import path format."""
        with pytest.raises(ValueError, match="Invalid import path format"):
            MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="invalid_path_no_colon",
                agent="also_invalid",
            )

    def test_simulated_user_must_be_subclass(self, tmp_path, mock_inner_agent_class):
        """Should raise TypeError if simulated user is not a SimulatedUser subclass."""

        class NotASimUser:
            pass

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [NotASimUser, mock_inner_agent_class]

            with pytest.raises(TypeError, match="must be a subclass of SimulatedUser"):
                MultiTurnAgent(
                    logs_dir=tmp_path / "logs",
                    simulated_user="test:NotASimUser",
                    agent="test:MockAgent",
                )

    def test_kwargs_parsing_from_json_string(self, tmp_path, mock_inner_agent_class):
        """Should parse kwargs from JSON string."""
        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [ScriptedSimUser, mock_inner_agent_class]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs='{"messages": ["hello", "world"]}',
            )

            assert agent._simulated_user.messages == ["hello", "world"]

    def test_kwargs_parsing_from_dict(self, tmp_path, mock_inner_agent_class):
        """Should accept kwargs as dict."""
        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [ScriptedSimUser, mock_inner_agent_class]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["hello"]},
            )

            assert agent._simulated_user.messages == ["hello"]

    def test_kwargs_rejects_non_object_json(self, tmp_path, mock_inner_agent_class):
        """Should reject JSON arrays or other non-object types."""
        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [ScriptedSimUser, mock_inner_agent_class]

            with pytest.raises(ValueError, match="Expected JSON object"):
                MultiTurnAgent(
                    logs_dir=tmp_path / "logs",
                    simulated_user="test:SimUser",
                    agent="test:Agent",
                    simulated_user_kwargs='["not", "a", "dict"]',
                )

    @pytest.mark.asyncio
    async def test_setup_calls_both_agents(
        self, tmp_path, mock_environment, mock_inner_agent_class
    ):
        """Setup should call setup on both inner agent and simulated user."""
        setup_called = {"sim_user": False, "inner": False}

        class TrackingSimUser(SimulatedUser):
            async def setup(self, env):
                setup_called["sim_user"] = True

            async def next_message(self, conv):
                raise SimulatedUserDone()

        class TrackingInnerAgent:
            def __init__(self, logs_dir, **kwargs):
                self.logs_dir = logs_dir
                self.logs_dir.mkdir(parents=True, exist_ok=True)

            async def setup(self, env):
                setup_called["inner"] = True

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [TrackingSimUser, TrackingInnerAgent]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
            )

            await agent.setup(mock_environment)

            assert setup_called["sim_user"] is True
            assert setup_called["inner"] is True

    @pytest.mark.asyncio
    async def test_run_conversation_loop(self, tmp_path, mock_environment):
        """Run should execute conversation loop until SimulatedUserDone."""
        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [ScriptedSimUser, MockHarborInnerAgent]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["hello", "how are you", "bye"]},
            )

            await agent.setup(mock_environment)
            await agent.run("initial instruction", mock_environment, MagicMock())

            # Should have processed all 3 messages - check conversation history
            history = agent.conversation_history
            assert len(history) == 6  # 3 user + 3 agent messages
            assert history[1]["content"] == "Response to: hello"
            assert history[3]["content"] == "Response to: how are you"
            assert history[5]["content"] == "Response to: bye"

    @pytest.mark.asyncio
    async def test_conversation_history_tracked(self, tmp_path, mock_environment):
        """Conversation history should be tracked correctly."""
        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [ScriptedSimUser, MockHarborInnerAgent]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["first", "second"]},
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            history = agent.conversation_history
            assert len(history) == 4
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "first"
            assert history[1]["role"] == "assistant"
            assert history[1]["content"] == "Response to: first"
            assert history[2]["role"] == "user"
            assert history[2]["content"] == "second"
            assert history[3]["role"] == "assistant"
            assert history[3]["content"] == "Response to: second"

    @pytest.mark.asyncio
    async def test_max_turns_limit(self, tmp_path, mock_environment):
        """Should stop after max_turns even if simulated user continues."""

        class InfiniteSimUser(SimulatedUser):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.count = 0

            async def next_message(self, conv):
                self.count += 1
                return f"message {self.count}"

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [
                InfiniteSimUser,
                lambda logs_dir, **kw: MockHarborInnerAgent(
                    logs_dir, response_fn=lambda m: "ok", **kw
                ),
            ]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                max_turns=5,
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            # Should stop at max_turns
            assert len(agent.conversation_history) == 10  # 5 turns * 2 messages each

    @pytest.mark.asyncio
    async def test_last_response_in_history(self, tmp_path, mock_environment):
        """Last response from inner agent should be in conversation history."""
        call_count = {"value": 0}

        def counting_response(msg: str) -> str:
            call_count["value"] += 1
            return f"response {call_count['value']}"

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [
                ScriptedSimUser,
                lambda logs_dir, **kw: MockHarborInnerAgent(
                    logs_dir, response_fn=counting_response, **kw
                ),
            ]

            agent = MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["a", "b", "c"]},
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            # Last response should be in conversation history
            history = agent.conversation_history
            assert history[-1]["content"] == "response 3"

    @pytest.mark.asyncio
    async def test_agent_kwargs_passed(self, tmp_path, mock_environment):
        """Inner agent should receive kwargs."""
        received_kwargs = {}

        class MockInnerAgent:
            def __init__(self, logs_dir, **kwargs):
                self.logs_dir = logs_dir
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                received_kwargs.update(kwargs)

            async def setup(self, env):
                pass

            async def run(self, instruction: str, environment: Any, context: Any) -> None:
                pass

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [ScriptedSimUser, MockInnerAgent]

            MultiTurnAgent(
                logs_dir=tmp_path / "logs",
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["hello"]},
                agent_kwargs={"custom_option": "value", "another": 123},
            )

            assert received_kwargs["custom_option"] == "value"
            assert received_kwargs["another"] == 123

    @pytest.mark.asyncio
    async def test_trajectory_saved_after_run(self, tmp_path, mock_environment):
        """Should save trajectory.json after run completes."""
        import json

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [
                ScriptedSimUser,
                lambda logs_dir, **kw: MockHarborInnerAgent(
                    logs_dir, response_fn=lambda _: "ok", **kw
                ),
            ]

            logs_dir = tmp_path / "logs"
            agent = MultiTurnAgent(
                logs_dir=logs_dir,
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["hello", "world"]},
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            trajectory_path = logs_dir / "trajectory.json"
            assert trajectory_path.exists()

            trajectory = json.loads(trajectory_path.read_text())
            assert trajectory["schema_version"] == "ATIF-v1.5"
            assert trajectory["agent"]["name"] == "multi-turn-agent"
            assert trajectory["agent"]["version"] == "0.1.0"
            assert len(trajectory["steps"]) == 4
            assert trajectory["final_metrics"]["total_steps"] == 4

    @pytest.mark.asyncio
    async def test_trajectory_steps_structure(self, tmp_path, mock_environment):
        """Trajectory steps should have correct structure."""
        import json

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [
                ScriptedSimUser,
                lambda logs_dir, **kw: MockHarborInnerAgent(
                    logs_dir, response_fn=lambda _: "agent-response", **kw
                ),
            ]

            logs_dir = tmp_path / "logs"
            agent = MultiTurnAgent(
                logs_dir=logs_dir,
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["hello"]},
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            trajectory = json.loads((logs_dir / "trajectory.json").read_text())
            steps = trajectory["steps"]

            # First step: simulated user message
            assert steps[0]["step_id"] == 1
            assert steps[0]["source"] == "user"
            assert steps[0]["message"] == "hello"
            assert steps[0]["extra"]["simulated"] is True
            assert steps[0]["extra"]["turn"] == 0

            # Second step: agent response
            assert steps[1]["step_id"] == 2
            assert steps[1]["source"] == "agent"
            assert steps[1]["message"] == "agent-response"
            assert steps[1]["extra"]["turn"] == 0

    @pytest.mark.asyncio
    async def test_trajectory_includes_inner_agent_name(self, tmp_path, mock_environment):
        """Trajectory extra should include inner agent name."""
        import json

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [
                ScriptedSimUser,
                lambda logs_dir, **kw: MockHarborInnerAgent(
                    logs_dir, agent_name="custom-inner-agent", response_fn=lambda m: "ok", **kw
                ),
            ]

            logs_dir = tmp_path / "logs"
            agent = MultiTurnAgent(
                logs_dir=logs_dir,
                simulated_user="test:SimUser",
                agent="test:Agent",
                simulated_user_kwargs={"messages": ["hello"]},
                max_turns=10,
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            trajectory = json.loads((logs_dir / "trajectory.json").read_text())
            assert trajectory["extra"]["inner_agent"] == "custom-inner-agent"
            assert trajectory["extra"]["max_turns"] == 10

    @pytest.mark.asyncio
    async def test_trajectory_empty_when_immediate_done(self, tmp_path, mock_environment):
        """Trajectory should handle immediate SimulatedUserDone."""
        import json

        class ImmediateDoneUser(SimulatedUser):
            async def next_message(self, conv):
                raise SimulatedUserDone("Done immediately")

        with patch("harbor_agent.multi_turn.agent._import_class") as mock_import:
            mock_import.side_effect = [
                ImmediateDoneUser,
                lambda logs_dir, **kw: MockHarborInnerAgent(
                    logs_dir, response_fn=lambda m: "ok", **kw
                ),
            ]

            logs_dir = tmp_path / "logs"
            agent = MultiTurnAgent(
                logs_dir=logs_dir,
                simulated_user="test:SimUser",
                agent="test:Agent",
            )

            await agent.setup(mock_environment)
            await agent.run("instruction", mock_environment, MagicMock())

            trajectory = json.loads((logs_dir / "trajectory.json").read_text())
            assert len(trajectory["steps"]) == 0
            assert trajectory["final_metrics"]["total_steps"] == 0
