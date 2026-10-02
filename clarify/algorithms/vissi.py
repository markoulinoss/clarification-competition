# SPDX-FileCopyrightText: 2026 Markos Balaskas <t8230101@aueb.gr>
#
# SPDX-License-Identifier: MIT

"""VissiClarifier.

Work-in-progress clarification algorithm. The LLM is asked for a solution to
the coding problem. If it abstains and raises a question instead, the question
is forwarded to the user and the LLM answers with the clarification in context.

Team: Vissi
Team Members: Markos Balaskas
Main Contact: t8230101@aueb.gr
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

PROMPT_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

If the problem is ambiguous in a way that blocks a correct solution, instead ask
exactly one short clarifying question and nothing else.
Otherwise, enclose your solution in ```python and ```.
""".strip()


class VissiClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {}

    def __init__(self, config=None):
        super().__init__({**self.DEFAULT_CONFIG, **(config or {})})

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
                    "{entry_point}", problem["entry_point"]
                ),
            }
        ]

        response = env.llm(messages)
        messages += [{"role": "assistant", "content": response}]

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # No valid implementation in the response, treat it as a clarifying question.
                try:
                    answer = env.ask_human(response)
                    messages += [{"role": "user", "content": answer}]
                    response = env.llm(messages)
                    messages += [{"role": "assistant", "content": response}]
                except TooManyQuestionException:
                    return response
