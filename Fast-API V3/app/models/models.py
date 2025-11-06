from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any
from datetime import datetime

class Criteria(BaseModel):
    name: str = Field(..., min_length=2, example="Vue.js", description="Name of the evaluation criterion")
    weight: int = Field(ge=1, le=10, example=9, description="Importance weight from 1 to 10")

class CriteriaSet(BaseModel):
    technical_skills: List[Criteria]
    experience: List[Criteria]
    soft_skills: List[Criteria]

class JobAnalysisRequest(BaseModel):
    job_title: str
    job_description: str

class CandidateRequest(BaseModel):
    job_title: str
    job_description: str
    top_n: int = 5
    temperature: float = 0.2

class QueryRequest(BaseModel):
    custom_query: str
    top_n: int = 5

class CandidateRanking(BaseModel):
    name: str
    match_score: float = Field(..., ge=0, le=10)
    reasons: List[str]
    gaps: List[str]
    highlights: List[str]

class LLMRankingResponse(BaseModel):
    ranked_candidates: List[CandidateRanking]
    faiss_candidates: Optional[list] = None

class ContactInfo(BaseModel):
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    email: Optional[str] = None

class Candidate(BaseModel):
    name: str
    relevance_score: int
    matching_points: List[str]
    relevant_experience: str
    contact: Optional[ContactInfo] = None

class LLMQueryResponse(BaseModel):
    query: str
    candidates: List[Candidate]
    faiss_candidates: Optional[List[str]] = []

class RankedCandidate(BaseModel):
    candidate_name: str
    technical_score: float
    experience_score: float
    soft_skills_score: float
    total_score: float
    strengths: List[str]
    weaknesses: List[str]
    justification: str

class LLMRankingResponse_1(BaseModel):
    ranking: List[RankedCandidate]
    summary: str
    faiss_candidates: List[str]

class JOBNLQ_Request(BaseModel):
    query: str

class JobDescription(BaseModel):
    title: str
    summary: str
    responsibilities: List[str]
    requirements: List[str]
    experience_level: str
    technologies: List[str]

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None

class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message to the agent")
    session_id: str = Field(..., description="Chat session identifier")
    current_jd: JobDescription = Field(..., description="Current job description to optimize")
    auto_approve: bool = Field(default=False, description="Auto-approve agent actions without human confirmation")

# FIXED: Added missing fields
class AgentActionData(BaseModel):
    query: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    modifications: Optional[Dict[str, Any]] = None
    questions: Optional[List[str]] = None
    reasoning: Optional[str] = None  # This was missing in your response
    summary: Optional[str] = None

# FIXED: Added missing pending_action_id field
class AgentChatResponse(BaseModel):
    message: str = Field(..., description="Agent response message")
    action_type: str = Field(..., description="Type of action taken by agent")
    action_data: Optional[AgentActionData] = None
    requires_approval: bool = Field(default=False, description="Whether action requires human approval")
    updated_jd: Optional[JobDescription] = None
    session_id: str = Field(..., description="Chat session identifier")
    conversation_summary: Optional[Dict[str, Any]] = None
    pending_action_id: Optional[str] = None  # ADDED: This was missing

class SessionResetRequest(BaseModel):
    session_id: str = Field(..., description="Session to reset")

# FIXED: Added missing pending_actions field
class SessionStatusResponse(BaseModel):
    session_id: str
    active: bool
    message_count: int
    last_activity: Optional[str] = None
    pending_actions: int = Field(default=0, description="Number of pending actions")  # ADDED

# ADDED: New model for approval requests
class ActionApprovalRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    action_id: str = Field(..., description="Pending action ID")
    approved: bool = Field(..., description="Whether to approve the action")
    current_jd: JobDescription = Field(..., description="Current job description")

# ADDED: Response model for approval
class ActionApprovalResponse(BaseModel):
    session_id: str
    action_id: str
    approved: bool
    message: str
    action_type: str
    action_data: Optional[AgentActionData] = None
    updated_jd: Optional[JobDescription] = None
    conversation_summary: Optional[Dict[str, Any]] = None