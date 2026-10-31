"""Mock model adapter for testing and demonstrating the evaluation system.

This adapter is not a Model Culi implementation. Its outputs are deterministic
test data and must never be presented as actual Model Culi intelligence scores.
"""

from __future__ import annotations


class MockModelAdapter:
    name = "mock-model"
    version = "0.0.0-mock"

    _responses = {
        "Dịch sang tiếng Anh: 'Xin chào'": "Hello",
        "Dịch sang tiếng Anh: 'Cảm ơn'": "Thank you",
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Write a Rust function that returns the sum of two i32 values.": "fn sum(a: i32, b: i32) -> i32 { a + b }",
        "In Dioxus, which macro is commonly used to define a component's UI?": "The rsx! macro is commonly used.",
        "If A > B and B > C, is A > C?": "Yes",
        "Plan the steps to compile and test a Rust project.": "Run cargo build, then cargo test.",
        "You need to run Rust tests. Which tool should you use?": "Use cargo test.",
        "Explain why memory matters for intelligence.": "Memory supports continuity and reasoning.",
    }

    def generate(self, prompt: str) -> str:
        return self._responses.get(prompt, "")
