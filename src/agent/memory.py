from dataclasses import dataclass, field


@dataclass
class AgentState:

    question: str

    messages: list[dict] = field(
        default_factory=list
    )

    tool_calls: list[dict] = field(
        default_factory=list
    )

    tool_results: list[dict] = field(
        default_factory=list
    )

    final_answer: str = ""

    error: str | None = None

    def add_message(
        self,
        message: dict
    ) -> None:

        self.messages.append(
            message
        )

    def add_tool_call(
        self,
        tool_call: dict
    ) -> None:

        self.tool_calls.append(
            tool_call
        )

    def add_tool_result(
        self,
        tool_result: dict
    ) -> None:

        self.tool_results.append(
            tool_result
        )

    def set_final_answer(
        self,
        answer: str
    ) -> None:

        self.final_answer = answer

    def set_error(
        self,
        error: str
    ) -> None:

        self.error = error