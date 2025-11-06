from fastapi import APIRouter, HTTPException,Depends
from app.models.models import JobAnalysisRequest, CriteriaSet,CandidateRequest,LLMRankingResponse,LLMQueryResponse,QueryRequest,LLMRankingResponse_1,JOBNLQ_Request,JobDescription
from app.services.llm_service import LLMService
from app.services.cv_matcher import CVMatcherService
from app.services.cv_matcher_via_uploads import MatcherViaUploads
from app.services.pdf_service import PDFService
from fastapi import FastAPI, File, UploadFile, Form,HTTPException
from typing import List
import json
from fastapi.responses import JSONResponse
import logging
from app.config import Config
from pydantic import ValidationError


router = APIRouter(tags=["Job Analysis"])
llm_service = LLMService(api_key=Config.GROQ_API_KEY)

def get_cv_matcher():
    return CVMatcherService()
async def get_upload_matcher():
    return MatcherViaUploads()

@router.post("/analyze-job", response_model=CriteriaSet)
async def analyze_job(request: JobAnalysisRequest):
    try:
        return await llm_service.extract_criteria(request.dict())
    except TimeoutError:
        raise HTTPException(504, "LLM timeout")
    except ValidationError as e:
        raise HTTPException(422, f"Validation error: {str(e)}")
    
@router.post("/generate-jd", response_model=JobDescription)
async def generate_jd(request: JOBNLQ_Request):
    try:
        return await llm_service.generate_job_description(request.query)
    except TimeoutError:
        raise HTTPException(504, "LLM timeout")
    except ValidationError as e:
        raise HTTPException(422, f"Validation error: {str(e)}")




@router.post("/top-candidates", response_model=LLMRankingResponse)
async def get_top_candidates(
    request: CandidateRequest,
    service: CVMatcherService = Depends(get_cv_matcher)
):
    try:
        return  service.get_top_candidates_with_llm(
            job_title=request.job_title,
            job_description=request.job_description,
            top_n=request.top_n,
            temperature=request.temperature
        )
    except TimeoutError:
        raise HTTPException(504, "LLM timeout")
    except ValidationError as e:
        raise HTTPException(422, f"Validation error: {str(e)}")


@router.post("/query-search", response_model=LLMQueryResponse)
async def get_top_candidates(
    request: QueryRequest,
    service: CVMatcherService = Depends(get_cv_matcher)
):
    try: 
        return service.run_custom_query(
            custom_query=request.custom_query,
            top_n=request.top_n,
        )
    except TimeoutError:
        raise HTTPException(504, "LLM timeout")
    except ValidationError as e:
        raise HTTPException(422, f"Validation error: {str(e)}")



@router.post("/rank")
async def rank_candidates(
    files: List[UploadFile] = File(...),
    criteria_json: str = Form(...),
    job_title : str = Form(...),
    job_descreption : str = Form(...),
    matcher: MatcherViaUploads = Depends(get_upload_matcher)
):
    try:
        # Process criteria
        criteria_data = json.loads(criteria_json)
        criteria = CriteriaSet.parse_obj(criteria_data)
        
        # Process uploaded files first
        await matcher.process_uploads(files)
        
        # Get ranking with both FAISS and LLM
        result =  matcher.get_top_candidates_with_llm(
            job_title=job_title,
            job_description=job_descreption,
            criteria=criteria,
            top_n=3
        )
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for criteria")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    