import os
import faiss
import numpy as np
import pickle
import time
import json
from app.models.models import LLMRankingResponse,LLMQueryResponse
from fastapi import HTTPException
from pydantic import ValidationError
from tqdm import tqdm
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from app.config import Config
from app.prompts.prompt_templates import build_ranking_prompt
from app.prompts.prompt_templates import build_custom_query_prompt

import logging
import pandas as pd
import json
from typing import Any
from pydantic import ValidationError
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cv_matcher')

class CVMatcherService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.groq_llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL
        )
        self.index = None
        self.metadata = None
        self._load_index()
    
    # @property
    # def embedding_model(self):
    #     if self._embedding_model is None:
    #         logger.info("Loading sentence transformer model...")
    #         self._embedding_model = SentenceTransformer(self.embedding_model_name)
    #     return self._embedding_model

    def _load_index(self):
        if Config.FAISS_INDEX_PATH.exists() and Config.FAISS_METADATA_PATH.exists():
            print(Config.FAISS_INDEX_PATH)
            self.index = faiss.read_index(str(Config.FAISS_INDEX_PATH))
            with open(Config.FAISS_METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)

    def transcribe_csv(self, csv_path: str) -> Dict[str, Dict]:
        """Extracts text and metadata from CSV with French columns"""
        try:
            # Read CSV with tab separator
            df = pd.read_csv(csv_path, sep=None, engine='python')
        except Exception as e:
            logger.error(f"Could not read CSV: {e}")
            raise ValueError(f"Could not read CSV: {e}")

        # Validate required columns
        required_columns = ["Nom du CV et profil", "Texte du CV"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        cv_docs = []
        for i, row in df.iterrows():
            if pd.isnull(row["Nom du CV et profil"]) or pd.isnull(row["Texte du CV"]):
                continue

            # Clean and normalize text
            text = ' '.join(str(row["Texte du CV"]).split())

            doc = {
                "text": text,
                "metadata": {
                    "filename": str(row["Nom du CV et profil"]),
                    "source": "csv",
                    "row_id": i
                }
            }
            cv_docs.append(doc)

        logger.info(f"Extracted {len(cv_docs)} CVs from CSV")
        return cv_docs

    def chunk_documents(self, documents: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        chunked_docs = []
        for doc in tqdm(documents, desc="Chunking documents"):
            text = doc["text"]
            chunks = splitter.split_text(text)
            chunks = [chunk for chunk in chunks if len(chunk) > 50]

            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                }
                chunked_docs.append(chunked_doc)

        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    def create_faiss_index(self, chunked_docs: List[Dict]) -> Dict:
        """Create or update FAISS index with new documents"""
        texts = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]

        logger.info("Generating embeddings...")
        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        faiss.normalize_L2(embeddings)
        
        # Create new index or merge with existing
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            cv_mapping = {}
        else:
            cv_mapping = self.metadata["cv_mapping"]

        self.index.add(embeddings)

        # Update metadata
        for i, metadata in enumerate(metadatas):
            name = metadata["filename"]
            if name not in cv_mapping:
                cv_mapping[name] = []
            cv_mapping[name].append(i)

        self.metadata = {
            "document_count": len(chunked_docs),
            "unique_candidates": len(cv_mapping),
            "cv_mapping": cv_mapping,
            "created_at": time.time(),
            "chunk_texts": texts,
            "chunk_metadata": metadatas
        }

        # Save updated index and metadata
        faiss.write_index(self.index,Config.FAISS_INDEX_PATH)
        with open(Config.FAISS_METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

        return {
            "status": "success",
            "index_path": Config.FAISS_INDEX_PATH,
            "cv_count": len(cv_mapping),
            "chunk_count": len(chunked_docs)
        }
      
    
    def build_cv_database(self, csv_path: str, force_rebuild: bool = False) -> Dict:
        """Build or update CV database"""
        start_time = time.time()
        if not force_rebuild and self.load_index():
            return {
                "status": "loaded_existing",
                "cv_count": self.metadata["unique_candidates"],
                "chunk_count": self.metadata["document_count"]
            }

        logger.info("Building new FAISS index...")
        cv_docs = self.transcribe_csv(csv_path)
        chunked_docs = self.chunk_documents(cv_docs)
        processing_time = time.time() - start_time
        print(f"Database built in {processing_time:.2f} seconds")
        return self.create_faiss_index(chunked_docs)


    def get_top_candidates_with_faiss(self, job_query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top candidates using existing index"""
        if not self.index or not self.metadata:
            raise ValueError("Index not loaded. Call build_cv_database() first.")

        query_embedding = self.embedding_model.encode([job_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        k = top_n * 5
        distances, indices = self.index.search(query_embedding, k)

        candidate_scores = {}
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            for candidate, chunk_indices in self.metadata["cv_mapping"].items():
                if idx in chunk_indices:
                    score = float(distance)
                    candidate_scores[candidate] = candidate_scores.get(candidate, 0) + score
                    break

        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_candidate_contexts(self, candidate_names: List[str], max_chunks_per_candidate: int = 5) -> Dict[str, List[str]]:
        """Get contexts from loaded metadata"""
        if not self.metadata:
            raise ValueError("Metadata not loaded. Call build_cv_database() first.")

        candidate_contexts  = {}
        chunk_texts = self.metadata["chunk_texts"]
        for candidate in candidate_names:
            if candidate in self.metadata["cv_mapping"]:
                # Get indices of chunks for this candidate
                chunk_indices = self.metadata["cv_mapping"][candidate]

                # # Use a subset if there are too many chunks
                if len(chunk_indices) > max_chunks_per_candidate:
                    chunk_indices = chunk_indices[:max_chunks_per_candidate]

                # Get text chunks
                contexts = [chunk_texts[idx] for idx in chunk_indices]
                candidate_contexts[candidate] = contexts

        return candidate_contexts

    def get_top_candidates_with_llm(self,job_title: str, job_description: str,
                                 top_n: int = 5, temperature: float = 0.2) -> LLMRankingResponse:
        """Search for top candidates using FAISS + LLM integration."""
        start_time = time.time()

        # Create job query
        job_query = f"Job Title: {job_title}\nJob Description: {job_description}"

        # First get top candidates using FAISS search
        logger.info("Retrieving top candidates using FAISS...")
        search_k = top_n * 2  # Get more candidates than needed for diversity
        top_candidates = self.get_top_candidates_with_faiss(job_query, search_k)
        print(f"Top candidate {top_candidates}")

        # Extract just the candidate names
        candidate_names = [name for name, score in top_candidates]
        logger.info(f"Top candidates from FAISS: {candidate_names}")

        # Get context chunks for these candidates
        candidate_contexts = self.get_candidate_contexts(candidate_names)
        print(f"candidate contexts {candidate_contexts}")


        # Create context string for LLM
        all_contexts = []
        for candidate, contexts in candidate_contexts.items():
            candidate_text = f"--- Candidate: {candidate} ---\n"
            candidate_text += "\n".join(contexts)
            all_contexts.append(candidate_text)

        context = "\n\n".join(all_contexts)
        print(f"the context {context}")

        # # Create the prompt
        # prompt_text = f"""
        # As an expert HR recruiter, your task is to rank candidates for a job position based on their CVs.

        # Job Position: {job_title}

        # Job Requirements:
        # {job_description}

        # Below are excerpts from candidate CVs. Analyze each candidate's qualifications, experience, and skills
        # to determine how well they match the job requirements.

        # {context}

        # Please provide:
        # 1. A ranked list of the top {top_n} candidates in order of suitability
        # 2. For each candidate, explain why they are a good match and any potential gaps
        # 3. Include specific qualifications, skills, or experiences from their CV that match the job requirements

        # Your response should be well-structured and easy to read.
        # """
        prompt_text = build_ranking_prompt(job_title, job_description, context)
        # Call LLM directly
        logger.info("Querying LLM for candidate ranking...")
        result = self.groq_llm.predict(prompt_text)
        print(result)
        processing_time = time.time() - start_time
        logger.info(f"LLM ranking completed in {processing_time:.2f} seconds")
        # Extract and validate the result
        api_result = self._extract_json(result,"search")
        api_result["faiss_candidates"]= candidate_names

        return api_result
        
    def run_custom_query(self, custom_query: str, top_n: int = 5) -> LLMQueryResponse:
        """Run a custom query against the CV database using FAISS + LLM."""
        start_time = time.time()

        # First get top candidates using FAISS search
        logger.info(f"Retrieving candidates for query: {custom_query[:50]}...")
        search_k = top_n * 2  # Get more candidates than needed for diversity
        top_candidates = self.get_top_candidates_with_faiss(custom_query, search_k)

        # Extract just the candidate names
        candidate_names = [name for name, score in top_candidates]
        logger.info(f"Top candidates from FAISS: {candidate_names}")

        # Get context chunks for these candidates
        candidate_contexts = self.get_candidate_contexts(candidate_names)
        print(f"candidate contexts {candidate_contexts}")

        # Create context string for LLM
        all_contexts = []
        for candidate, contexts in candidate_contexts.items():
            candidate_text = f"--- Candidate: {candidate} ---\n"
            candidate_text += "\n".join(contexts)
            all_contexts.append(candidate_text)

        context = "\n\n".join(all_contexts)
        print(context)

        # # Create the prompt
        # prompt_text = f"""
        # You are an expert at analyzing resumes and CVs. Based on the following search query:

        # "{custom_query}"

        # Analyze these CV excerpts and determine which candidates best match the requirements:

        # {context}

        # Please provide:
        # 1. A ranked list of the top {top_n} candidates in order of suitability for this query
        # 2. For each candidate, explain why they match the query criteria
        # 3. Include specific qualifications, skills, or experiences from their CV that are relevant

        # Your response should be well-structured and easy to read.
        # """

        
        prompt_text = build_custom_query_prompt(custom_query,context,top_n)
        # Call LLM directly
        logger.info(f"Running custom query: {custom_query[:50]}...")
        result = self.groq_llm.predict(prompt_text)
        print(result)

        processing_time = time.time() - start_time
        logger.info(f"Custom query completed in {processing_time:.2f} seconds")
        api_result = self._extract_json(result,"query")
        api_result["faiss_candidates"]= candidate_names

        return api_result

    
    def _extract_json(self, text: str, type: str) -> dict:
        """Extract and validate JSON from LLM output"""
        try:
            # Try to extract from ```json ... ```
            json_str = text.split("```json")[1].split("```")[0].strip()
        except IndexError:
            # Fallback: extract from first '{' to last '}'
            json_str = text[text.find("{"):text.rfind("}")+1]

        try:
            # Load JSON string into Python dict
            raw_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from LLM response: {e}\nRaw string: {json_str}")

        try:
            # Validate and convert to your API model
            if type=="search":
                validated = LLMRankingResponse(**raw_json)
            else:
                validated = LLMQueryResponse(**raw_json)
        except ValidationError as ve:
            raise ValueError(f"LLM response does not match expected format: {ve}")

        # Return as dict for API response
        return validated.dict()