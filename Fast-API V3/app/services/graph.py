from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os 
from langchain_core.messages import SystemMessage
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# from .llm import llm_model
from langgraph.prebuilt import ToolNode, tools_condition

from .context_manager import trim_messages
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

class State(TypedDict):
    messages: Annotated[list, add_messages]


os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

llm_model = init_chat_model("groq:llama3-8b-8192")
tavily_tool  = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)
tools = [tavily_tool]

# LLM that knows about the tools
llm_with_tools = llm_model.bind_tools(tools)


def chatbot(state: State):

    # Trim context
    state["messages"] = trim_messages(state["messages"], max_tokens=6000)

    # Add the system message to the conversation context
    system_message = SystemMessage(content="""
You are a helpful assistant that optimizes job descriptions (JDs) based on conversations with the user.

Context:
- You have memory of the current conversation and can use it to understand the user’s needs.
- You only ask follow-up questions if the user input is ambiguous or missing key information required to optimize the JD.
- If the user’s message suggests a need for up-to-date trends, labor market data, or external facts (e.g. salary benchmarks, job market trends, tech popularity), you will trigger a web search using the Tavily search tool.
- only Once the user's needs are clarified (either directly or through previous messages), you modify the current JD to align with those needs and return the optimized version.

Rules:
- Speak naturally, like a human, and keep things clear and focused.
- Ask questions only when necessary for clarity or optimization context.
- Do not optimize a JD unless user needs are clearly stated or inferred from prior conversation.
- Only trigger a web search if the input contains elements that require external information (e.g. 'what’s trending', 'popular tools in 2025', etc.).
- Don’t repeat the user’s input unnecessarily.
- note :only in case you have returned a new updated JD cadre the new jd with <optimized-jd></optimized-jd>
""")

    full_messages = [system_message] + state["messages"]

    return {"messages": [llm_with_tools.invoke(full_messages)]}

# ----------------------------------------
# 4. Build the graph
# ----------------------------------------

graph_builder = StateGraph(State)

# Add chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Add routing logic: if tools are needed, go to "tools", else END
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # auto-detects tool calls
    {"tools": "tools", END: END}
)

# Loop back from tool node to chatbot
graph_builder.add_edge("tools", "chatbot")

# Entry point
graph_builder.add_edge(START, "chatbot")

# Compile graph with memory
from langgraph.checkpoint.memory import MemorySaver
graph = graph_builder.compile(checkpointer=MemorySaver())

# from dotenv import load_dotenv
# from typing import Annotated, Literal
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
# from langchain.chat_models import init_chat_model
# from pydantic import BaseModel, Field
# from typing_extensions import TypedDict
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.messages import AIMessage, HumanMessage
# import json
# import os

# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("jd_agent")

# load_dotenv()
# os.environ["TAVILY_API_KEY"] = "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"
# os.environ["GROQ_API_KEY"] = "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"

# # 🔸 LLM + Tools
# llm_model = init_chat_model("groq:llama3-8b-8192")
# tavily_search = TavilySearchResults(
#     max_results=3,
#     search_depth="advanced",
#     include_answer=True,
#     include_raw_content=True
# )

# # -----------------------------
# # Define State
# # -----------------------------
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     jd: str
#     message_type: str | None
#     next: str | None

# # -----------------------------
# # Message Type Classifier
# # -----------------------------
# class MessageClassifier(BaseModel):
#     message_type: Literal["clarifier", "searcher", "optimizer"] = Field(...)

# def classify_message(state: State):
#     classifier_llm = llm_model.with_structured_output(MessageClassifier)
#     last_message = state["messages"][-1]
    
#     # Get conversation history for better classification
#     conversation_context = ""
#     if len(state["messages"]) > 1:
#         recent_messages = state["messages"][-5:]  # Last 5 messages for context
#         conversation_context = "\n".join([
#             f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
#             for msg in recent_messages[:-1]
#         ])

#     prompt = [
#         {"role": "system", "content": """You are a classifier that decides the next best agent to handle the request.

# Rules for classification:
# - "clarifier": User is asking questions about requirements, needs clarification on what to optimize, or wants to discuss preferences
# - "searcher": User wants market research, industry trends, salary info, or competitor analysis
# - "optimizer": User wants to optimize/rewrite the JD, or is ready for the final output

# Consider the conversation history to make better decisions."""},
#         {"role": "user", "content": f"""
# Job description:
# {state['jd']}

# Recent conversation:
# {conversation_context}

# Current user message: "{last_message.content}"

# What type of agent should handle this?"""}
#     ]
    
#     result = classifier_llm.invoke(prompt)
#     logger.info(f"[Classifier] Message classified as: {result.message_type}")
#     return {"message_type": result.message_type}

# # -----------------------------
# # Router Node
# # -----------------------------
# def router(state: State):
#     logger.info(f"[Router] Routing to agent: {state.get('message_type')}")
#     return {"next": state.get("message_type", "optimizer")}

# # -----------------------------
# # Clarifier Agent
# # -----------------------------
# def clarifier_agent(state: State):
#     logger.info("[Agent] Clarifier activated.")
#     last_message = state["messages"][-1]
    
#     # Build conversation history for context
#     conversation_history = []
#     for msg in state["messages"][:-1]:  # All messages except the last one
#         role = "user" if isinstance(msg, HumanMessage) else "assistant"
#         conversation_history.append({"role": role, "content": msg.content})
    
#     messages = [
#         {"role": "system", "content": """
# You are a JD optimization assistant. Your role is to ask targeted questions to clarify what the user wants to improve in their job description.

# Based on the conversation history, ask follow-up questions about:
# - Ideal candidate profile and experience level
# - Work setting preferences (remote, hybrid, onsite)
# - Required vs nice-to-have skills
# - Company culture and values to highlight
# - Compensation and benefits approach
# - Tone and style preferences

# Don't optimize the JD yet - just gather requirements and preferences.
# """},
#         *conversation_history,
#         {"role": "user", "content": last_message.content}
#     ]
    
#     response = llm_model.invoke(messages)
#     return {"messages": [AIMessage(content=response.content)]}

# # -----------------------------
# # Searcher Agent
# # -----------------------------
# def search_agent(state: State):
#     logger.info("[Agent] Searcher activated.")
#     jd = state['jd']
#     last_message = state["messages"][-1]
    
#     # Create search query based on user request and JD
#     query = f"job market trends hiring insights {last_message.content} {jd[:200]}"
    
#     try:
#         results = tavily_search.invoke({"query": query})
        
#         # Format results with sources
#         formatted_results = []
#         for result in results[:3]:
#             formatted_results.append(
#                 f"• {result['content'][:300]}...\n  📍 Source: {result['url']}"
#             )
        
#         sources = "\n\n".join(formatted_results)
        
#         response_content = f"""🔍 **Market Research Results:**

# {sources}

# 💡 **Key Takeaways:**
# Based on current trends, consider incorporating these insights into your job description optimization."""
        
#     except Exception as e:
#         logger.error(f"Search error: {e}")
#         response_content = "I encountered an issue with the search. Let me help you optimize the JD based on best practices instead."
    
#     return {
#         "messages": [AIMessage(content=response_content)]
#     }

# # -----------------------------
# # Optimizer Agent
# # -----------------------------
# def optimizer_agent(state: State):
#     logger.info("[Agent] Optimizer activated.")
#     jd = state['jd']
    
#     # Build conversation context
#     conversation_context = ""
#     user_preferences = []
    
#     for msg in state["messages"][:-1]:
#         if isinstance(msg, HumanMessage):
#             user_preferences.append(msg.content)
#         conversation_context += f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}\n"
    
#     preferences_summary = "\n".join([f"- {pref}" for pref in user_preferences[-3:]])  # Last 3 user inputs
    
#     messages = [
#         {"role": "system", "content": f"""
# You are a professional job description optimizer with expertise in modern hiring practices.

# CONTEXT FROM CONVERSATION:
# {conversation_context}

# USER PREFERENCES:
# {preferences_summary}

# Your task:
# 1. Rewrite the JD to be clear, engaging, and inclusive
# 2. Incorporate user preferences from the conversation
# 3. Use modern, appealing language that attracts top talent
# 4. Structure it professionally with clear sections
# 5. Highlight company benefits and culture
# 6. Make requirements realistic and well-prioritized

# Format as a well-structured job description, not JSON. Use markdown formatting for better readability.
# """},
#         {"role": "user", "content": f"Original Job Description to optimize:\n\n{jd}"}
#     ]
    
#     response = llm_model.invoke(messages)
#     return {"messages": [AIMessage(content=response.content)]}

# # -----------------------------
# # Build the Graph
# # -----------------------------
# graph_builder = StateGraph(State)
# graph_builder.add_node("classifier", classify_message)
# graph_builder.add_node("router", router)
# graph_builder.add_node("clarifier", clarifier_agent)
# graph_builder.add_node("searcher", search_agent)
# graph_builder.add_node("optimizer", optimizer_agent)

# graph_builder.add_edge(START, "classifier")
# graph_builder.add_edge("classifier", "router")
# graph_builder.add_conditional_edges("router", lambda s: s.get("next"), {
#     "clarifier": "clarifier",
#     "searcher": "searcher", 
#     "optimizer": "optimizer",
# })
# graph_builder.add_edge("clarifier", END)
# graph_builder.add_edge("searcher", END)
# graph_builder.add_edge("optimizer", END)

# # -----------------------------
# # Compile With MemorySaver
# # -----------------------------
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)

















# from dotenv import load_dotenv
# from typing import Annotated, Literal
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
# from langchain.chat_models import init_chat_model
# from pydantic import BaseModel, Field
# from typing_extensions import TypedDict
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.messages import AIMessage, HumanMessage
# import json
# import os

# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("jd_agent")

# load_dotenv()
# os.environ["TAVILY_API_KEY"] = "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"
# os.environ["GROQ_API_KEY"] = "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"

# # 🔸 LLM + Tools
# llm_model = init_chat_model("groq:llama3-8b-8192")
# tavily_search = TavilySearchResults(
#     max_results=3,
#     search_depth="advanced",
#     include_answer=True,
#     include_raw_content=True
# )

# # -----------------------------
# # Define State
# # -----------------------------
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     jd: str
#     message_type: str | None
#     next: str | None

# # -----------------------------
# # Message Type Classifier
# # -----------------------------
# class MessageClassifier(BaseModel):
#     message_type: Literal["clarifier", "searcher", "optimizer"] = Field(...)

# def classify_message(state: State):
#     classifier_llm = llm_model.with_structured_output(MessageClassifier)
#     last_message = state["messages"][-1]

#     prompt = [
#         {"role": "system", "content": "You are a classifier that decides the next best agent to handle the request."},
#         {"role": "user", "content": f"""
# Given the job description:
# {state['jd']}

# User's latest message: \"{last_message.content}\"
# """}
#     ]
#     result = classifier_llm.invoke(prompt)
#     logger.info(f"[Classifier] Message classified as: {result.message_type}")
#     return {"message_type": result.message_type}

# # -----------------------------
# # Router Node
# # -----------------------------
# def router(state: State):
#     logger.info(f"[Router] Routing to agent: {state.get("message_type")}")
#     return {"next": state.get("message_type", "optimizer")}

# # -----------------------------
# # Clarifier Agent
# # -----------------------------
# def clarifier_agent(state: State):
#     logger.info("[Agent] Clarifier activated.")
#     last_message = state["messages"][-1]
#     messages = [
#         {"role": "system", "content": """
# You are a JD optimization assistant. Ask targeted questions to clarify what the user wants to improve in their job description.
# Don't optimize yet, just gather needs. Your goal is to help tailor the JD according to their ideal candidate, work setting (remote, hybrid), technologies, tone, and selling points.
# """},
#         {"role": "user", "content": last_message.content}
#     ]
#     response = llm_model.invoke(messages)
#     return {"messages": [AIMessage(content=response.content)]}

# # -----------------------------
# # Searcher Agent
# # -----------------------------
# def search_agent(state: State):
#     logger.info("[Agent] Searcher activated.")
#     jd = state['jd']
#     query = f"Recent job trends and keyword insights for: {jd}"
#     results = tavily_search.invoke({"query": query})

#     # Format results with sources
#     formatted_results = []
#     for result in results[:3]:
#         formatted_results.append(
#             f"• {result['content']}\n  Source: {result['url']}"
#         )
    
#     sources = "\n\n".join(formatted_results)
    
#     return {
#         "messages": [AIMessage(
#             content=f"🔍 Here's what I found about current job market trends:\n\n{sources}"
#         )]
#     }

# # -----------------------------
# # Optimizer Agent
# # -----------------------------
# def optimizer_agent(state: State):
#     logger.info("[Agent] Optimizer activated.")
#     jd = state['jd']
#     prompt = [
#         {"role": "system", "content": """
# You are a professional job description optimizer.
# Your job is to:
# - Rewrite the JD clearly and concisely
# - Use inclusive and engaging language
# - Highlight benefits, values, remote work (if applicable), and clarity in tasks
# - Emphasize any user preferences from the conversation
# - Incorporate relevant keywords from market trends

# Format your output as JSON with the following structure:
# {
#     "title": "...",
#     "summary": "...",
#     "responsibilities": ["...", "..."],
#     "requirements": ["...", "..."],
#     "benefits": ["...", "..."]
# }
# """},
#         {"role": "user", "content": f"Here is the original JD:\n{jd}"}
#     ] + state["messages"]  # Let memory handle history automatically
#     response = llm_model.invoke(prompt)
#     return {"messages": [AIMessage(content=response.content)]}

# # -----------------------------
# # Build the Graph
# # -----------------------------
# graph_builder = StateGraph(State)
# graph_builder.add_node("classifier", classify_message)
# graph_builder.add_node("router", router)
# graph_builder.add_node("clarifier", clarifier_agent)
# graph_builder.add_node("searcher", search_agent)
# graph_builder.add_node("optimizer", optimizer_agent)

# graph_builder.add_edge(START, "classifier")
# graph_builder.add_edge("classifier", "router")
# graph_builder.add_conditional_edges("router", lambda s: s.get("next"), {
#     "clarifier": "clarifier",
#     "searcher": "searcher",
#     "optimizer": "optimizer",
# })
# graph_builder.add_edge("clarifier", END)
# graph_builder.add_edge("searcher", END)
# graph_builder.add_edge("optimizer", END)

# # -----------------------------
# # Compile With MemorySaver
# # -----------------------------
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
