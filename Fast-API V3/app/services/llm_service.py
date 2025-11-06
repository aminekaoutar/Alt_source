import json
import logging
from groq import Groq
from pydantic import ValidationError
from app.models.models import CriteriaSet  ,JobDescription
from app.prompts.prompt_templates import build_dynamic_prompt ,build_jd_generation_prompt
from langdetect import detect, LangDetectException
from typing import Optional, List, Dict




logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b-8192"

    async def extract_criteria(self, request_data: dict) -> CriteriaSet:
        """Extrait et valide les critères de sélection"""
        prompt = build_dynamic_prompt(
            request_data['job_title'], 
            request_data['job_description']
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=2000
            )
            
            json_response = self._extract_json(response.choices[0].message.content)
            # normalized_data = self._normalize_weights(json_response)
            return CriteriaSet(**json_response)
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Erreur de traitement : {str(e)}")
            raise ValueError(f"Erreur de format : {str(e)}")

    async def generate_job_description(self, job_query: str) -> JobDescription:
        """Génère une fiche de poste à partir d'une requête NLQ"""
        prompt = build_jd_generation_prompt(job_query)

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.5,
                max_tokens=1500
            )

            json_response = self._extract_json(response.choices[0].message.content)
            return JobDescription(**json_response)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Erreur de génération de fiche de poste : {str(e)}")
            raise ValueError(f"Erreur de format : {str(e)}")
 
    def _extract_json(self, text: str) -> dict:
            """Extrait le JSON de la réponse LLM"""
            try:
                # Gestion des backticks JSON
                json_str = text.split('```json')[1].split('```')[0].strip()
            except IndexError:
                # Fallback si formatage absent
                json_str = text[text.find('{'):text.rfind('}')+1]
            
            return json.loads(json_str)