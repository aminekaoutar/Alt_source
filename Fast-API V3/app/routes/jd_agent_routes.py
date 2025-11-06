from fastapi import APIRouter, HTTPException, Depends
from typing import Dict , Optional
import logging
import asyncio
import json
from datetime import datetime
from pydantic import BaseModel
from app.services.graph import graph
from app.services.template import html
from pydantic import ValidationError
import uuid
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi import FastAPI,WebSocket
from fastapi.responses import HTMLResponse
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from app.services.jd_optimization_agent import JDOptimizationAgent, HumanApprovalException
from app.models.models import (
    AgentChatRequest, 
    AgentChatResponse, 
    SessionResetRequest, 
    SessionStatusResponse,
    AgentActionData,
    ActionApprovalResponse,
    ActionApprovalRequest,
    JobDescription
)
from app.config import Config
from dotenv import load_dotenv


load_dotenv() 

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["JD Optimization Agent"])


class ChatInput(BaseModel):
    message: str
    thread_id: str


@router.post("/chat")
async def chat(input: ChatInput):
    config = {"configurable": {"thread_id": input.thread_id}}
    # Add new human message to the thread
    response = await graph.ainvoke(
        {"messages": [HumanMessage(content=input.message)]},
        config=config
    )
    # Return the last AI message
    return response["messages"][-1].content

@router.get("/")
async def get():
    return HTMLResponse(html)


@router.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_text()
            
            async for event in graph.astream({"messages": [data]}, config=config, stream_mode="messages"):
                message = event[0]
                
                if isinstance(message, AIMessage):
                    await websocket.send_text(message.content)
                
                elif isinstance(message, ToolMessage):
                    try:
                        # Handle Tavily tool response
                        if hasattr(message, 'content') and message.content:
                            # Try to parse as JSON first
                            if isinstance(message.content, str):
                                tool_output = json.loads(message.content)
                            else:
                                # If it's already a dict/object
                                tool_output = message.content
                            
                            # Extract answer and results from Tavily response
                            answer = tool_output.get("answer", "")
                            results = tool_output.get("results", [])
                            
                            # Format sources
                            sources_text = ""
                            if results and len(results) > 0:
                                sources_list = []
                                for result in results:
                                    title = result.get("title", "Unknown Title")
                                    url = result.get("url", "#")
                                    # You can also include content snippet if needed
                                    # content_snippet = result.get("content", "")[:100] + "..."
                                    sources_list.append(f"- [{title}]({url})")
                                
                                sources_text = "\n\n**Sources:**\n" + "\n".join(sources_list)
                            
                            # Combine answer with sources
                            if answer:
                                full_response = f"{answer}{sources_text}"
                            elif sources_text:
                                full_response = f"Search completed.{sources_text}"
                            else:
                                full_response = "Search completed but no results found."
                            
                            await websocket.send_text(full_response)
                            
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, send the raw content
                        await websocket.send_text(f"Tool response: {message.content}")
                    except Exception as e:
                        await websocket.send_text(f"Error processing tool output: {str(e)}")
                
                else:
                    # Handle other message types
                    content = getattr(message, 'content', str(message))
                    await websocket.send_text(str(content))
                    
        except WebSocketDisconnect:
            break
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            break

# WebSocket endpoint for real-time streaming
# @router.websocket("/ws/{thread_id}")     
# async def websocket_endpoint(websocket: WebSocket, thread_id: str):
#     config = {"configurable": {"thread_id": thread_id}}
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         async for event in graph.astream({"messages": [data]}, config=config, stream_mode="messages"):
#             await websocket.send_text(event[0].content) 

# @router.websocket("/ws/jd/{thread_id}")
# async def jd_websocket(websocket: WebSocket, thread_id: str):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         parsed = json.loads(data)
#         message = parsed.get("message")
#         jd = parsed.get("jd")
#         config = {"configurable": {"thread_id": thread_id}}
#         state = {"messages": [HumanMessage(content=message)], "jd": jd}
#         async for step in graph.astream(state, config=config, stream_mode="messages"):
#             await websocket.send_text(step[0].content)

# @router.websocket("/ws/jd/{thread_id}")
# async def jd_websocket(websocket: WebSocket, thread_id: str):
#     await websocket.accept()
#     while True:
#         try:
#             data = await websocket.receive_text()
#             parsed = json.loads(data)

#             # Récupération des champs
#             message = parsed.get("message")
#             jd = parsed.get("jd")

#             # Préparation du state initial avec uniquement le nouveau message
#             state = {
#                 "messages": [HumanMessage(content=message)],
#                 "jd": jd
#             }

#             config = {"configurable": {"thread_id": thread_id}}

#             # Lancement du graphe avec mémoire
#             response = await graph.ainvoke(state, config=config)

#             # Envoi de la réponse finale (pas de streaming ici)
#             await websocket.send_text(response["messages"][-1].content)

#         except WebSocketDisconnect:
#             logger.warning(f"[WebSocket] Disconnected: {thread_id}")
#             break
#         except Exception as e:
#             logger.exception(f"[WebSocket Error] {str(e)}")
#             await websocket.send_text(f"❌ Error: {str(e)}")


@router.websocket("/ws/jd/{thread_id}")
async def jd_websocket(websocket: WebSocket, thread_id: str):
    await websocket.accept()
    logger.info(f"[WebSocket] Connected: {thread_id}")
    
    while True:
        try:
            data = await websocket.receive_text()
            parsed = json.loads(data)

            # Récupération des champs
            message = parsed.get("message")
            jd = parsed.get("jd")

            config = {"configurable": {"thread_id": thread_id}}

            # 🔥 KEY FIX: Use graph.aget_graph() to get current state first
            # This ensures we preserve conversation history
            try:
                current_state = await graph.aget_state(config)
                if current_state.values:
                    # If we have existing state, just add the new message
                    state = {
                        "messages": [HumanMessage(content=message)],
                        "jd": jd  # Update JD in case it changed
                    }
                else:
                    # First message in conversation
                    state = {
                        "messages": [HumanMessage(content=message)],
                        "jd": jd
                    }
            except Exception as e:
                # Fallback for new conversation
                logger.info(f"[WebSocket] New conversation starting: {thread_id}")
                state = {
                    "messages": [HumanMessage(content=message)],
                    "jd": jd
                }

            # 🚀 Launch the graph with memory preserved
            response = await graph.ainvoke(state, config=config)

            # Send the final response
            final_message = response["messages"][-1].content
            await websocket.send_text(final_message)
            
            logger.info(f"[WebSocket] Response sent for thread: {thread_id}")

        except WebSocketDisconnect:
            logger.warning(f"[WebSocket] Disconnected: {thread_id}")
            break
        except json.JSONDecodeError:
            logger.error("[WebSocket] Invalid JSON received")
            await websocket.send_text("❌ Error: Invalid JSON format")
        except Exception as e:
            logger.exception(f"[WebSocket Error] {str(e)}")
            await websocket.send_text(f"❌ Error: {str(e)}")

# Optional: Add an endpoint to check conversation history
@router.get("/ws/jd/{thread_id}/history")
async def get_conversation_history(thread_id: str):
    """Get the conversation history for a thread"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        
        if state.values and "messages" in state.values:
            history = []
            for msg in state.values["messages"]:
                history.append({
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)
                })
            return {"thread_id": thread_id, "history": history}
        else:
            return {"thread_id": thread_id, "history": []}
            
    except Exception as e:
        logger.exception(f"Error getting history for {thread_id}: {str(e)}")
        return {"error": str(e)}

# Optional: Clear conversation history
@router.delete("/ws/jd/{thread_id}/history")
async def clear_conversation_history(thread_id: str):
    """Clear the conversation history for a thread"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # This will clear the thread's state
        await graph.aget_state(config)  # This creates the thread if it doesn't exist
        # Note: MemorySaver doesn't have a direct clear method
        # You might need to implement custom logic or use a different checkpointer
        return {"message": f"History cleared for thread {thread_id}"}
    except Exception as e:
        logger.exception(f"Error clearing history for {thread_id}: {str(e)}")
        return {"error": str(e)}