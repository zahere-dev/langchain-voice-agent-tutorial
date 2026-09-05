"""A web-research tool the agent can call before writing a Notion page."""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from .. import config
from ..logging_utils import log_stage

_llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.3)
_search = TavilySearch(tavily_api_key=config.TAVILY_API_KEY, max_results=5)


@tool
def research_topic(topic: str) -> str:
    """Research a topic on the web and return a concise markdown summary.

    Use this before creating a Notion page whenever the user asks you to
    "research", "look into", or "find out about" something. The summary
    should then be passed as the content when creating the Notion page.
    """
    log_stage("researcher -> Tavily", input=topic)
    raw_results = _search.invoke({"query": topic})
    results_text = "\n".join(
        f"- {r['title']}: {r['content']}" for r in raw_results.get("results", [])
    )
    log_stage("Tavily -> researcher", output=results_text)

    prompt = (
        "You are a research assistant. Using the raw search results below, "
        "write a concise, well-organized summary about the topic. Use short "
        "headings and bullet points. Keep it under 300 words.\n\n"
        f"Topic: {topic}\n\nRaw search results:\n{results_text}"
    )
    response = _llm.invoke(prompt)
    log_stage("summarize LLM -> researcher", output=response.content)
    return response.content
