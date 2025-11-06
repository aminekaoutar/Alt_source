# app/services/jd_optimization_agent.py

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from groq import Groq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, ValidationError
import uuid
import os
from app.models.models import JobDescription
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
load_dotenv()


class ActionType(Enum):
    SEARCH = "search"
    OPTIMIZE = "optimize"
    CLARIFY = "clarify"
    FINALIZE = "finalize"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class PendingAction:
    """Represents an action waiting for approval"""
    id: str
    action_type: ActionType
    action_data: Dict[str, Any]
    reasoning: str
    created_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING

@dataclass
class SearchAction:
    query: str
    reasoning: str

@dataclass
class OptimizationAction:
    modifications: Dict[str, Any]
    reasoning: str

@dataclass
class ClarificationAction:
    questions: List[str]
    reasoning: str

class AgentResponse(BaseModel):
    message: str
    action_type: ActionType
    action_data: Optional[Dict[str, Any]] = None
    requires_approval: bool = False
    updated_jd: Optional[JobDescription] = None
    pending_action_id: Optional[str] = None  # Add this for tracking pending actions

class HumanApprovalException(Exception):
    """Raised when human approval is required but not granted"""
    pass

class JDOptimizationAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        os.environ["TAVILY_API_KEY"] = "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"
        self.tavily_search = TavilySearch(
            max_results=3,
            topic="general",
            include_answer=True,
            search_depth="advanced",
            
        )
        self.llm_service = LLMService(groq_api_key)
        self.model = "llama3-8b-8192"
        self.conversation_history = []
        self.pending_actions: Dict[str, PendingAction] = {}  # Store pending actions
        
    def _build_system_prompt(self) -> str:
        return """You are an expert HR consultant and job description optimization specialist. 
        Your role is to help users improve their job descriptions by:
        
        1. **Analysis**: Analyzing current job descriptions for gaps, inconsistencies, or improvements
        2. **Research**: Using web search to find current market trends, salary data, and best practices
        3. **Optimization**: Suggesting specific improvements based on industry standards and research
        4. **Clarification**: Asking targeted questions to better understand user needs
        
        Available actions:
        - SEARCH: Research industry trends, salary data, competitor JDs, or best practices
        - OPTIMIZE: Suggest specific modifications to the job description
        - CLARIFY: Ask questions to better understand requirements
        - FINALIZE: Present the optimized job description
        
        Always provide clear reasoning for your suggestions and be ready to explain your decisions.
        When suggesting significant changes, explain the benefits and potential impact.
        
        Respond in JSON format with:
        {
            "message": "Your response to the user",
            "action_type": "search|optimize|clarify|finalize", 
            "action_data": {
                // For SEARCH: {"query": "search query"}
                // For OPTIMIZE: {"modifications": {...}}
                // For CLARIFY: {"questions": [...]}
                // For FINALIZE: {"summary": "..."}
            },
            "requires_approval": true/false,
            "reasoning": "Why you chose this action"
        }"""

    async def process_user_input(
        self, 
        user_input: str, 
        current_jd: JobDescription,
        auto_approve: bool = False
    ) -> AgentResponse:
        """Process user input and determine next action"""
        
        # Add user input to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": user_input,
            "jd_context": current_jd.dict()
        })
        
        # Build the prompt with context
        prompt = self._build_conversation_prompt(user_input, current_jd)
        
        try:
            # Get agent decision
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1500
            )
            
            agent_decision = self._extract_json(response.choices[0].message.content)
            logger.info(f"Agent decision: {agent_decision}")
            
            # Execute the decided action
            return await self._execute_action(agent_decision, current_jd, auto_approve)
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return AgentResponse(
                message=f"I encountered an error while processing your request. Could you please rephrase or try again?",
                action_type=ActionType.CLARIFY,
                requires_approval=False
            )

    def _build_conversation_prompt(self, user_input: str, current_jd: JobDescription) -> str:
        """Build conversation prompt with context"""
        jd_summary = f"""
        Current Job Description:
        - Title: {current_jd.title}
        - Summary: {current_jd.summary}
        - Experience Level: {current_jd.experience_level}
        - Key Responsibilities: {len(current_jd.responsibilities)} items
        - Required Skills: {len(current_jd.requirements)} items
        - Technologies: {len(current_jd.technologies)} items
        """
        
        conversation_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            conversation_context = "Recent conversation:\n" + "\n".join([
                f"- {msg['role']}: {msg['content'][:100]}..." 
                for msg in recent_history
            ])
        
        return f"""
        {jd_summary}
        
        {conversation_context}
        
        User's current request: {user_input}
        
        Based on this context, what action should I take to help optimize this job description?
        Consider whether I need to:
        1. Search for current market data or trends
        2. Optimize specific sections based on best practices  
        3. Ask clarifying questions
        4. Finalize the improvements
        
        Respond with your decision and reasoning.
        """

    async def _execute_action(
        self, 
        agent_decision: Dict, 
        current_jd: JobDescription,
        auto_approve: bool = False
    ) -> AgentResponse:
        """Execute the agent's decided action"""
        
        try:
            action_type = ActionType(agent_decision["action_type"])
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid action_type in agent_decision: {agent_decision}")
            return self._create_fallback_response()
        
        # Ensure action_data exists
        action_data = agent_decision.get("action_data", {})
        requires_approval = agent_decision.get("requires_approval", False)
        
        # If approval is required and auto_approve is False, create pending action
        if requires_approval and not auto_approve:
            return self._create_pending_action(agent_decision, current_jd)
        
        # Execute action directly
        if action_type == ActionType.SEARCH:
            return await self._execute_search_action(agent_decision, auto_approve)
            
        elif action_type == ActionType.OPTIMIZE:
            return await self._execute_optimization_action(agent_decision, current_jd, auto_approve)
            
        elif action_type == ActionType.CLARIFY:
            return self._execute_clarification_action(agent_decision)
            
        elif action_type == ActionType.FINALIZE:
            return await self._execute_finalization_action(agent_decision, current_jd)
            
        else:
            logger.error(f"Unknown action type: {action_type}")
            return self._create_fallback_response()

    def _create_pending_action(self, agent_decision: Dict, current_jd: JobDescription) -> AgentResponse:
        """Create a pending action that requires approval"""
        from datetime import datetime
        
        action_id = str(uuid.uuid4())
        action_type = ActionType(agent_decision["action_type"])
        
        pending_action = PendingAction(
            id=action_id,
            action_type=action_type,
            action_data=agent_decision.get("action_data", {}),
            reasoning=agent_decision.get("reasoning", ""),
            created_at=datetime.now().isoformat(),
            status=ApprovalStatus.PENDING
        )
        
        self.pending_actions[action_id] = pending_action
        
        # Create appropriate message based on action type
        if action_type == ActionType.SEARCH:
            query = pending_action.action_data.get("query", "")
            message = f"I'd like to search for: '{query}' to help optimize your job description. {pending_action.reasoning}"
        elif action_type == ActionType.OPTIMIZE:
            modifications = pending_action.action_data.get("modifications", {})
            mod_summary = self._summarize_changes(modifications)
            message = f"I'd like to make the following optimizations: {mod_summary}. {pending_action.reasoning}"
        else:
            message = f"I have a {action_type.value} action that requires your approval. {pending_action.reasoning}"
        
        return AgentResponse(
            message=f"{message}\n\nPlease use the approval endpoint to approve or reject this action.",
            action_type=action_type,
            action_data=pending_action.action_data,
            requires_approval=True,
            pending_action_id=action_id
        )

    async def approve_action(self, action_id: str, approved: bool, current_jd: JobDescription) -> AgentResponse:
        """Process approval for a pending action"""
        
        if action_id not in self.pending_actions:
            raise ValueError(f"Action {action_id} not found")
        
        pending_action = self.pending_actions[action_id]
        
        if not approved:
            pending_action.status = ApprovalStatus.REJECTED
            return AgentResponse(
                message="Action rejected. How else can I help you optimize this job description?",
                action_type=ActionType.CLARIFY,
                requires_approval=False
            )
        
        # Approve and execute the action
        pending_action.status = ApprovalStatus.APPROVED
        
        # Reconstruct the agent decision format
        agent_decision = {
            "action_type": pending_action.action_type.value,
            "action_data": pending_action.action_data,
            "reasoning": pending_action.reasoning,
            "requires_approval": False  # Already approved
        }
        
        # Execute the approved action
        if pending_action.action_type == ActionType.SEARCH:
            return await self._execute_search_action(agent_decision, auto_approve=True)
        elif pending_action.action_type == ActionType.OPTIMIZE:
            return await self._execute_optimization_action(agent_decision, current_jd, auto_approve=True)
        else:
            return AgentResponse(
                message="Action approved and executed successfully.",
                action_type=pending_action.action_type,
                requires_approval=False
            )

    def _create_fallback_response(self) -> AgentResponse:
        """Create a fallback response when something goes wrong"""
        return AgentResponse(
            message="To better help you optimize this job description, could you tell me what specific aspects you'd like me to focus on? For example: improving the requirements, making it more attractive to candidates, updating the technologies, or enhancing the job summary?",
            action_type=ActionType.CLARIFY,
            action_data={
                "questions": [
                    "What specific aspects of the job description would you like me to improve?",
                    "Are you looking to attract more candidates or improve the quality of applicants?",
                    "Do you want me to research current market trends for this role?"
                ],
                "reasoning": "Need clarification to provide better assistance"
            },
            requires_approval=False
        )

    async def _execute_search_action(
        self, 
        agent_decision: Dict, 
        auto_approve: bool = False
    ) -> AgentResponse:
        """Execute web search"""
        
        action_data = agent_decision.get("action_data", {})
        search_query = action_data.get("query", "")
        reasoning = agent_decision.get("reasoning", "")
        
        if not search_query:
            logger.error("No search query provided in action_data")
            return self._create_fallback_response()
        
        try:
            # Execute search
            search_results = self.tavily_search.invoke({"query": search_query})
            
            # Process search results
            processed_results = self._process_search_results(search_results)
            
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Searched for: {search_query}",
                "search_results": processed_results
            })
            
            return AgentResponse(
                message=f"I found some relevant information about '{search_query}'. {processed_results['summary']}",
                action_type=ActionType.SEARCH,
                action_data={
                    "query": search_query,
                    "results": processed_results,
                    "reasoning": reasoning
                },
                requires_approval=False
            )
            
        except Exception as e:
            logger.error(f"Search execution failed: {str(e)}")
            return AgentResponse(
                message="I encountered an issue while searching. Let me try a different approach or ask you for more specific information.",
                action_type=ActionType.CLARIFY,
                requires_approval=False
            )

    async def _execute_optimization_action(
        self, 
        agent_decision: Dict, 
        current_jd: JobDescription,
        auto_approve: bool = False
    ) -> AgentResponse:
        """Execute JD optimization"""
        
        action_data = agent_decision.get("action_data", {})
        modifications = action_data.get("modifications", {})
        reasoning = agent_decision.get("reasoning", "")
        
        if not modifications:
            logger.error("No modifications provided in action_data")
            return self._create_fallback_response()
        
        try:
            # Apply modifications to create updated JD
            updated_jd = self._apply_modifications(current_jd, modifications)
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": f"Applied optimizations: {reasoning}",
                "modifications": modifications
            })
            
            return AgentResponse(
                message=f"I've optimized your job description based on {reasoning}. Here are the key changes I made: {self._summarize_changes(modifications)}",
                action_type=ActionType.OPTIMIZE,
                action_data={
                    "modifications": modifications,
                    "reasoning": reasoning
                },
                requires_approval=False,
                updated_jd=updated_jd
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return AgentResponse(
                message="I had trouble applying those optimizations. Could you help me understand what specific changes you'd like to see?",
                action_type=ActionType.CLARIFY,
                requires_approval=False
            )

    def _execute_clarification_action(self, agent_decision: Dict) -> AgentResponse:
        """Execute clarification action"""
        
        action_data = agent_decision.get("action_data", {})
        questions = action_data.get("questions", [])
        reasoning = agent_decision.get("reasoning", "")
        
        if not questions:
            # Provide default questions if none specified
            questions = [
                "What specific aspects of the job description would you like me to improve?",
                "Are there any particular requirements or technologies you'd like me to focus on?"
            ]
        
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Asked clarifying questions: {reasoning}",
            "questions": questions
        })
        
        questions_text = "\n".join([f"• {q}" for q in questions])
        
        return AgentResponse(
            message=f"To better help you optimize this job description, I have a few questions:\n\n{questions_text}",
            action_type=ActionType.CLARIFY,
            action_data={
                "questions": questions,
                "reasoning": reasoning
            },
            requires_approval=False
        )

    async def _execute_finalization_action(
        self, 
        agent_decision: Dict, 
        current_jd: JobDescription
    ) -> AgentResponse:
        """Execute finalization action"""
        
        action_data = agent_decision.get("action_data", {})
        summary = action_data.get("summary", "Job description optimization completed")
        
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Finalized optimization: {summary}"
        })
        
        return AgentResponse(
            message=f"Your job description has been optimized! Here's a summary of the improvements: {summary}",
            action_type=ActionType.FINALIZE,
            action_data={"summary": summary},
            requires_approval=False,
            updated_jd=current_jd
        )

    # Remove the _request_human_approval method since we're handling it asynchronously

    def _process_search_results(self, search_results: Dict) -> Dict:
        """Process and summarize search results"""
        
        if isinstance(search_results, str):
            search_data = json.loads(search_results)
        else:
            search_data = search_results
            
        # Extract key information
        results = search_data.get("results", [])
        answer = search_data.get("answer", "")
        
        # Create summary
        key_points = []
        for result in results[:3]:  # Top 3 results
            title = result.get("title", "")
            content = result.get("content", "")[:200] + "..."
            key_points.append(f"• {title}: {content}")
        
        summary = f"Found {len(results)} relevant sources. "
        if answer:
            summary += f"Key insight: {answer[:150]}..."
        
        return {
            "summary": summary,
            "key_points": key_points,
            "answer": answer,
            "source_count": len(results)
        }

    def _apply_modifications(self, current_jd: JobDescription, modifications: Dict) -> JobDescription:
        """Apply modifications to job description"""
        
        jd_dict = current_jd.dict()
        
        for field, new_value in modifications.items():
            if field in jd_dict:
                if isinstance(jd_dict[field], list) and isinstance(new_value, list):
                    # For list fields, merge or replace based on the modification type
                    if field.endswith("_add"):
                        base_field = field.replace("_add", "")
                        if base_field in jd_dict:
                            jd_dict[base_field].extend(new_value)
                    else:
                        jd_dict[field] = new_value
                else:
                    jd_dict[field] = new_value
        
        return JobDescription(**jd_dict)

    def _summarize_changes(self, modifications: Dict) -> str:
        """Create human-readable summary of changes"""
        
        changes = []
        for field, value in modifications.items():
            if isinstance(value, list):
                changes.append(f"Updated {field} ({len(value)} items)")
            else:
                changes.append(f"Modified {field}")
        
        return ", ".join(changes)

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in backticks first
            if '```json' in text:
                json_str = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text and '{' in text:
                # Handle case where JSON is in backticks without 'json' label
                json_str = text.split('```')[1].strip()
            else:
                # Fallback: find JSON object
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = text[start:end]
                else:
                    raise ValueError("No JSON found in text")
            
            parsed_json = json.loads(json_str)
            
            # Validate that required fields exist
            if "action_type" not in parsed_json:
                logger.warning(f"Missing action_type in parsed JSON: {parsed_json}")
                raise ValueError("Missing action_type")
                
            return parsed_json
            
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.error(f"Failed to extract JSON from response: {text}")
            logger.error(f"Error: {str(e)}")
            # Return a default response structure
            return {
                "message": "I need to better understand your request. Could you provide more details about what specific improvements you're looking for?",
                "action_type": "clarify",
                "action_data": {
                    "questions": [
                        "What specific aspects of the job description would you like me to improve?",
                        "Are you looking to make it more attractive to candidates?",
                        "Should I research current market trends for this role?"
                    ]
                },
                "requires_approval": False,
                "reasoning": "Failed to parse agent response, requesting clarification"
            }

    def get_pending_actions(self) -> Dict[str, PendingAction]:
        """Get all pending actions for this session"""
        return {k: v for k, v in self.pending_actions.items() if v.status == ApprovalStatus.PENDING}

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.pending_actions = {}

    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation history"""
        return {
            "total_exchanges": len(self.conversation_history),
            "pending_actions": len(self.get_pending_actions()),
            "recent_actions": [
                {"role": msg["role"], "action": msg.get("content", "")[:50] + "..."} 
                for msg in self.conversation_history[-5:]
            ]
        }