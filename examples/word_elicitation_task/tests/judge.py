#!/usr/bin/env python3
"""LLM judge for word elicitation task.

Evaluates the simulated user's prompts - did they ask for definitions,
or create scenarios for natural word usage?
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

TARGET_WORDS = ["petrichor", "defenestration", "sonder", "phosphenes", "apricity"]


def extract_conversation(trajectory_path: str) -> list[dict[str, str]]:
    """Extract user prompts and assistant responses from trajectory."""
    with open(trajectory_path) as f:
        data: dict[str, Any] = json.load(f)

    conversation: list[dict[str, str]] = []
    for step in data.get("steps", []):
        source = step.get("source")
        msg = step.get("message", "")

        if source == "user":
            conversation.append({"role": "user", "content": msg})
        elif source == "agent":
            # The agent message contains JSON logs with a "result" field
            # Parse each line to find the result message
            text = _extract_agent_result(msg)
            if text:
                conversation.append({"role": "assistant", "content": text})

    return conversation


def _extract_agent_result(msg: str) -> str | None:
    """Extract the result text from agent JSON logs."""
    for line in msg.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and "result" in parsed:
                return str(parsed["result"])
        except json.JSONDecodeError:
            continue
    return None


def find_word_in_text(text: str, word: str) -> bool:
    """Check if word appears in text (case-insensitive)."""
    return word.lower() in text.lower()


def judge_user_prompt(
    client: anthropic.Anthropic, word: str, user_prompt: str
) -> tuple[bool, str]:
    """Judge if the user's prompt asked for a definition or created a natural scenario."""
    prompt = f"""You are judging a word elicitation game. The player must get an AI to use the word "{word}" naturally, WITHOUT asking for definitions.

Player's prompt:
"{user_prompt}"

Did the player ask for a definition, vocabulary explanation, or "what's the word for X"? Or did they create a scenario where the word might naturally appear?

DEFINITION-SEEKING (bad):
- "What's the word for the smell after rain?"
- "Do you know what that phenomenon is called?"
- "Can you tell me about unusual words for..."
- "What causes that scent?" (fishing for the word)
- Describing a concept and asking the assistant to name/identify it

SCENARIO-CREATING (good):
- "Write a paragraph about walking in the garden after a storm"
- "Help me with a story set in medieval Prague"
- "I'm writing about the feeling of watching strangers"
- Asking for creative writing where the word might naturally appear

Respond with ONLY: GOOD or BAD"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.content[0].text.strip().upper()
    is_good = "GOOD" in result
    return is_good, result


def main() -> None:
    # Find trajectory file
    trajectory_path: str | None = None
    for path in ["/logs/agent/trajectory.json", "/logs/trajectory.json"]:
        if os.path.exists(path):
            trajectory_path = path
            break

    if not trajectory_path:
        print("No trajectory file found")
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("0.0")
        return

    conversation = extract_conversation(trajectory_path)
    if not conversation:
        print("No conversation found")
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write("0.0")
        return

    print(f"Extracted {len(conversation)} messages")

    # Combine all assistant responses
    all_responses = "\n".join(
        msg["content"] for msg in conversation if msg["role"] == "assistant"
    )

    # Initialize Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set, falling back to word count only")
        found = sum(1 for w in TARGET_WORDS if find_word_in_text(all_responses, w))
        score = found / len(TARGET_WORDS)
        with open("/logs/verifier/reward.txt", "w") as f:
            f.write(f"{score:.2f}")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # For each word, check if it appeared and if user prompts were good
    score = 0
    for word in TARGET_WORDS:
        # Check if word appears in any response
        if not find_word_in_text(all_responses, word):
            print(f"  {word}: NOT FOUND")
            continue

        # Find which turn elicited this word
        for i, msg in enumerate(conversation):
            if msg["role"] == "assistant" and find_word_in_text(msg["content"], word):
                # Find the preceding user prompt
                user_prompt = None
                for j in range(i - 1, -1, -1):
                    if conversation[j]["role"] == "user":
                        user_prompt = conversation[j]["content"]
                        break

                if user_prompt:
                    is_good, judgment = judge_user_prompt(client, word, user_prompt)
                    if is_good:
                        print(f"  {word}: GOOD PROMPT -> +1")
                        score += 1
                    else:
                        print(f"  {word}: BAD PROMPT (asked for definition)")
                else:
                    # No user prompt found, give benefit of doubt
                    print(f"  {word}: FOUND (no prompt to judge) -> +1")
                    score += 1
                break

    final_score = score / len(TARGET_WORDS)
    print(f"\nFinal score: {score}/{len(TARGET_WORDS)} = {final_score:.2f}")

    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(f"{final_score:.2f}")


if __name__ == "__main__":
    main()
