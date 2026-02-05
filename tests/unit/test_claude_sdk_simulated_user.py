"""Unit tests for ClaudeSdkSimulatedUser."""

import pytest

from harbor_agent.multi_turn import ConversationMessage, SimulatedUserDone
from harbor_agent.multi_turn.claude_sdk_simulated_user import ClaudeSdkSimulatedUser


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeAssistantMessage:
    def __init__(self, content):
        self.content = content


@pytest.mark.asyncio
async def test_prompt_includes_goal_and_history():
    recorded = {}

    async def fake_query(prompt, options=None):
        recorded["prompt"] = prompt
        recorded["options"] = options
        yield FakeAssistantMessage([FakeTextBlock("hello")])

    user = ClaudeSdkSimulatedUser(goal="Book a flight", query_fn=fake_query)
    conversation: list[ConversationMessage] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]

    response = await user.next_message(conversation)

    assert response == "hello"
    assert "User goal:\nBook a flight" in recorded["prompt"]
    assert "User: hi" in recorded["prompt"]
    assert "Assistant: hello!" in recorded["prompt"]
    assert recorded["options"]["max_turns"] == 1
    assert "<<DONE>>" in recorded["options"]["system_prompt"]


@pytest.mark.asyncio
async def test_done_token_raises():
    async def fake_query(prompt, options=None):
        yield FakeAssistantMessage([FakeTextBlock("<<DONE>>")])

    user = ClaudeSdkSimulatedUser(goal="Do something", query_fn=fake_query)

    with pytest.raises(SimulatedUserDone):
        await user.next_message([])


@pytest.mark.asyncio
async def test_text_blocks_are_joined():
    async def fake_query(prompt, options=None):
        yield FakeAssistantMessage([FakeTextBlock("Hello"), FakeTextBlock(" world")])

    user = ClaudeSdkSimulatedUser(goal="Say hello", query_fn=fake_query)

    response = await user.next_message([])
    assert response == "Hello world"


@pytest.mark.asyncio
async def test_set_instruction_sets_goal_when_empty():
    async def fake_query(prompt, options=None):
        yield FakeAssistantMessage([FakeTextBlock("response")])

    user = ClaudeSdkSimulatedUser(query_fn=fake_query)
    user.set_instruction("Task instruction as goal")

    response = await user.next_message([])
    assert response == "response"
    assert user._goal == "Task instruction as goal"


@pytest.mark.asyncio
async def test_set_instruction_does_not_override_existing_goal():
    async def fake_query(prompt, options=None):
        yield FakeAssistantMessage([FakeTextBlock("response")])

    user = ClaudeSdkSimulatedUser(goal="Original goal", query_fn=fake_query)
    user.set_instruction("New instruction")

    await user.next_message([])
    assert user._goal == "Original goal"


@pytest.mark.asyncio
async def test_next_message_raises_if_no_goal():
    async def fake_query(prompt, options=None):
        yield FakeAssistantMessage([FakeTextBlock("response")])

    user = ClaudeSdkSimulatedUser(query_fn=fake_query)

    with pytest.raises(ValueError, match="goal must be set"):
        await user.next_message([])
