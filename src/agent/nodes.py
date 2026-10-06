import json

import ollama

from src.agent.tools import CodeSearchTool
from src.agent.memory import AgentState


class CodeToolCallingAgent:

    def __init__(
        self,
        model: str = "qwen3:4b"
    ):

        print("Initializing code tool-calling agent...")

        self.model = model

        self.code_search_tool = CodeSearchTool()

        self.tools = [
            self.code_search_tool.get_tool_definition()
        ]

        print("Code tool-calling agent initialized.")

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict
    ) -> str:

        try:

            if tool_name != CodeSearchTool.NAME:

                raise ValueError(
                    f"Unknown tool: {tool_name}"
                )

            results = self.code_search_tool.execute(
                query=arguments["query"],
                top_k=arguments.get("top_k", 5)
            )

            return json.dumps(
                {
                    "success": True,
                    "results": results,
                },
                indent=2,
                default=str
            )

        except Exception as error:

            return json.dumps(
                {
                    "success": False,
                    "error": type(error).__name__,
                    "message": str(error),
                },
                indent=2
            )

    # --------------------------------------------------
    # NODE 1 — LLM NODE
    # --------------------------------------------------

    def llm_node(
        self,
        state: AgentState
    ) -> AgentState:

        response = ollama.chat(
            model=self.model,
            messages=state.messages,
            tools=self.tools,
        )

        state.add_message(
            response["message"]
        )

        return state

    # --------------------------------------------------
    # NODE 2 — TOOL NODE
    # --------------------------------------------------

    def tool_node(
        self,
        state: AgentState
    ) -> AgentState:

        message = state.messages[-1]

        tool_calls = message.get(
            "tool_calls",
            []
        )

        for tool_call in tool_calls:

            function = tool_call["function"]

            tool_name = function["name"]

            arguments = function.get(
                "arguments",
                {}
            )

            state.add_tool_call(
                tool_call
            )

            tool_result = self.execute_tool(
                tool_name=tool_name,
                arguments=arguments,
            )

            state.add_tool_result(
                {
                    "tool_name": tool_name,
                    "result": tool_result,
                }
            )

            state.add_message(
                {
                    "role": "tool",
                    "content": tool_result,
                }
            )

        return state

    # --------------------------------------------------
    # NODE 3 — INITIALIZE STATE
    # --------------------------------------------------

    def initialize_state(
        self,
        question: str
    ) -> AgentState:

        state = AgentState(
            question=question
        )

        state.add_message(
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that answers "
                    "questions about a software codebase. "
                    "Use the code search tool whenever you "
                    "need information from the codebase. "
                    "Do not invent codebase details. "
                    "If a tool reports an error, explain the "
                    "problem clearly instead of inventing "
                    "tool results."
                ),
            }
        )

        state.add_message(
            {
                "role": "user",
                "content": question,
            }
        )

        return state

    # --------------------------------------------------
    # ROUTING
    # --------------------------------------------------

    def has_tool_calls(
        self,
        state: AgentState
    ) -> bool:

        if not state.messages:

            return False

        latest_message = state.messages[-1]

        return bool(
            latest_message.get(
                "tool_calls"
            )
        )

    # --------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------

    def set_final_answer(
        self,
        state: AgentState
    ) -> AgentState:

        if not state.messages:

            state.set_error(
                "No messages available."
            )

            return state

        latest_message = state.messages[-1]

        if latest_message.get("tool_calls"):

            return state

        answer = latest_message.get(
            "content",
            ""
        )

        if not answer:

            state.set_error(
                "LLM returned an empty answer."
            )

            return state

        state.set_final_answer(
            answer
        )

        return state

    # --------------------------------------------------
    # MAIN EXECUTION
    # --------------------------------------------------

    def ask(
        self,
        question: str
    ) -> str:

        state = self.initialize_state(
            question=question
        )

        state = self.llm_node(
            state
        )

        while self.has_tool_calls(
            state
        ):

            state = self.tool_node(
                state
            )

            state = self.llm_node(
                state
            )

        state = self.set_final_answer(
            state
        )

        if state.error:

            return (
                f"Agent error: {state.error}"
            )

        return state.final_answer