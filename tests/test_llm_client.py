"""Tool-calling LLM clients, exercised with fake provider SDK objects."""
import json
from types import SimpleNamespace as NS

from src.agent.llm_client import (GroqToolLLM, LLMReply, LLMUnavailable, OllamaToolLLM,
                                  ToolCall, assistant_message)


def _groq_response(tool_calls=None, content="", pt=11, ct=7):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=pt, completion_tokens=ct))


def _tc(id, name, args):
    return NS(id=id, function=NS(name=name, arguments=args))


class FakeGroqClient:
    def __init__(self, behaviors):
        self.behaviors, self.models_tried = list(behaviors), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.models_tried.append(kwargs["model"])
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def test_groq_parses_tool_calls_and_usage():
    client = FakeGroqClient([_groq_response([_tc("a", "read_file", json.dumps({"path": "x.py"}))])])
    reply = GroqToolLLM(model="m1", fallback_model="m2", client=client).chat([], [], 100)
    assert reply.tool_calls == [ToolCall("a", "read_file", {"path": "x.py"})]
    assert (reply.prompt_tokens, reply.completion_tokens) == (11, 7)


def test_groq_invalid_json_arguments_are_flagged_not_raised():
    client = FakeGroqClient([_groq_response([_tc("a", "edit_file", "{not json")])])
    reply = GroqToolLLM(model="m1", client=client).chat([], [], 100)
    assert "_invalid_json" in reply.tool_calls[0].arguments


def test_groq_falls_back_to_second_model_on_failure():
    client = FakeGroqClient([RuntimeError("429 rate limited"), _groq_response(content="hi")])
    reply = GroqToolLLM(model="primary", fallback_model="backup", client=client).chat([], [], 100)
    assert client.models_tried == ["primary", "backup"] and reply.content == "hi"


def test_groq_raises_when_every_model_fails():
    client = FakeGroqClient([RuntimeError("down"), RuntimeError("also down")])
    try:
        GroqToolLLM(model="a", fallback_model="b", client=client).chat([], [], 100)
    except RuntimeError as exc:
        assert "also down" in str(exc)
    else:
        raise AssertionError("expected the last error to propagate")


def test_missing_api_key_raises_a_clear_error():
    import src.agent.llm_client as mod
    original = mod.settings
    mod.settings = NS(GROQ_API_KEY="", GROQ_MODEL="m", FALLBACK_MODEL="f")
    try:
        GroqToolLLM()
    except LLMUnavailable as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("expected LLMUnavailable")
    finally:
        mod.settings = original


def test_assistant_message_round_trips_tool_calls_as_json_strings():
    msg = assistant_message(LLMReply(content="", tool_calls=[ToolCall("1", "grep_code", {"pattern": "x"})]))
    fn = msg["tool_calls"][0]["function"]
    assert msg["role"] == "assistant" and fn["name"] == "grep_code"
    assert json.loads(fn["arguments"]) == {"pattern": "x"}


def test_ollama_converts_history_and_parses_dict_and_object_responses():
    seen = {}

    class FakeOllama:
        def chat(self, **kwargs):
            seen.update(kwargs)
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "run_tests", "arguments": {"command": "x"}}}]},
                "prompt_eval_count": 5, "eval_count": 3}

    history = [assistant_message(LLMReply(tool_calls=[ToolCall("1", "read_file", {"path": "a.py"})]))]
    reply = OllamaToolLLM(model="qwen3:4b", client=FakeOllama()).chat(history, [], 50)
    assert seen["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}  # dict, not str
    assert reply.tool_calls[0].name == "run_tests" and reply.tool_calls[0].arguments == {"command": "x"}
    assert (reply.prompt_tokens, reply.completion_tokens) == (5, 3)
