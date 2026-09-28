"""QA fixture for PLT-4618 / PLT-4619: a stock LangChain agent that makes real tool calls.

No Trase imports, no network, no LLM key: a scripted chat model drives the LangChain
agent loop, so the only thing under test is whether the platform records the
framework's tool calls on the run timeline (framework: langchain).
"""
import logging

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

log = logging.getLogger("qa-langchain-tool-events")


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    n = len(text.split())
    log.info("QA_TOOL word_count -> %d", n)
    return n


@tool
def reverse_text(text: str) -> str:
    """Reverse a piece of text."""
    r = text[::-1]
    log.info("QA_TOOL reverse_text -> %s", r)
    return r


class ScriptedChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            msg = AIMessage(content="", tool_calls=[{"name": "word_count", "id": "call_1", "type": "tool_call",
                                                      "args": {"text": "trase os third party agent"}}])
        elif len(results) == 1:
            msg = AIMessage(content="", tool_calls=[{"name": "reverse_text", "id": "call_2", "type": "tool_call",
                                                      "args": {"text": "KNOWN_TOOL_OUTPUT"}}])
        else:
            msg = AIMessage(content=f"words={results[0].content}; reversed={results[1].content}")
        return ChatResult(generations=[ChatGeneration(message=msg)])


# One module-level compiled graph, so topology inspection can find it (PLT-4619).
graph = create_agent(ScriptedChatModel(), tools=[word_count, reverse_text])


def run():
    result = graph.invoke({"messages": [{"role": "user", "content": "count and reverse"}]})
    answer = result["messages"][-1].content
    log.info("QA_RESULT %s", answer)
    return answer
