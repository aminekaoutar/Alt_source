import faiss
import numpy as np
from typing import List, Dict
from fastapi import UploadFile
from .pdf_service import PDFService 
import faiss
import numpy as np
from app.models.models import LLMRankingResponse,LLMQueryResponse,CriteriaSet,LLMRankingResponse_1
import time
from tqdm import tqdm
import json
from pydantic import ValidationError
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
# from langchain_deepseek import ChatDeepSeek
from app.config import Config
from app.prompts.prompt_templates import build_scoring_prompt,build_ranking_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter # Assuming PDFService is in same module


class MatcherViaUploads:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.groq_llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL
        )
        self.pdf_service = PDFService()
        self.temp_index = None
        self.temp_metadata = None

    async def process_uploads(self, files: List[UploadFile]) -> dict:
        """Process uploaded CVs with progress tracking"""
        processed = await self.pdf_service.process_uploaded_cvs(files)
        print(processed)
        
        # Filter valid documents with progress
        cv_docs = []
        valid_files = tqdm(
            [f for f in processed.items() if 'error' not in f[1]],
            desc="Processing valid CVs",
            unit="cv"
        )
        
        for filename, data in valid_files:
            cv_docs.append({
                "text": data['text'],
                "metadata": {
                    "filename": filename,
                    "source": "upload",
                    "row_id": len(cv_docs)
                }
            })
        
        chunked_docs = self._chunk_documents(cv_docs)
        self._create_temp_index(chunked_docs)
        
        return {
            "status": "ready",
            "candidate_count": len(cv_docs),
            "chunk_count": len(chunked_docs)
        }

    def _chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents with progress tracking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]

        )
        
        chunked_docs = []
        for doc in tqdm(documents, desc="Chunking CVs", unit="cv"):
            text = doc["text"]
            chunks = splitter.split_text(text)
            chunks = [chunk for chunk in chunks if len(chunk) > 50]
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                })
        return chunked_docs

    def _create_temp_index(self, chunked_docs: List[Dict]):
        """Create index with batch processing tracking"""
        texts = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]

        # Batch embeddings with progress
        batch_size = 32
        embeddings = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating embeddings",
            unit="batch"
        ):
            batch = texts[i:i+batch_size]
            embeddings.append(self.embedding_model.encode(batch, convert_to_numpy=True))
        
        embeddings = np.vstack(embeddings)
        faiss.normalize_L2(embeddings)

        # Create index
        d = embeddings.shape[1]
        self.temp_index = faiss.IndexFlatIP(d)
        self.temp_index.add(embeddings)

        # Build metadata mapping
        cv_mapping = {}
        for idx, metadata in tqdm(
            enumerate(metadatas),
            desc="Indexing metadata",
            total=len(metadatas)
        ):
            name = metadata["filename"]
            cv_mapping.setdefault(name, []).append(idx)

        self.temp_metadata = {
            "cv_mapping": cv_mapping,
            "chunk_texts": texts,
            "created_at": time.time(),
            "chunk_metadata": metadatas
        }

    
    def get_top_candidates_with_faiss(self, job_query: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top candidates using existing index"""
        if not self.temp_index or not self.temp_metadata:
            raise ValueError("Process uploads first")

        query_embedding = self.embedding_model.encode([job_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        k = top_n *2
        distances, indices = self.temp_index.search(query_embedding, k)

        candidate_scores = {}
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            for candidate, chunk_indices in self.temp_metadata["cv_mapping"].items():
                if idx in chunk_indices:
                    score = float(distance)
                    candidate_scores[candidate] = candidate_scores.get(candidate, 0) + score
                    break

        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


    def get_candidate_contexts(self, candidate_names: List[str], max_chunks_per_candidate: int = 9) -> Dict[str, List[str]]:
        """Get contexts from loaded metadata"""
        if not self.temp_metadata:
            raise ValueError("Metadata not loaded")

        candidate_contexts  = {}
        chunk_texts = self.temp_metadata["chunk_texts"]
        for candidate in candidate_names:
            if candidate in self.temp_metadata["cv_mapping"]:
                # Get indices of chunks for this candidate
                chunk_indices = self.temp_metadata["cv_mapping"][candidate]

                # Use a subset if there are too many chunks
                if len(chunk_indices) > max_chunks_per_candidate:
                    chunk_indices = chunk_indices[:max_chunks_per_candidate]

                # Get text chunks
                contexts = [chunk_texts[idx] for idx in chunk_indices]
                candidate_contexts[candidate] = contexts

        return candidate_contexts
    
    def format_candidate_contexts(self, candidate_contexts: Dict[str, List[str]], compact: bool = True) -> str:
        """Format candidate contexts in a more compact way"""
        all_contexts = []
        
        for candidate, contexts in candidate_contexts.items():
            candidate_text = f"--- Candidate: {candidate} ---\n"
            
            if compact:
                # Join all contexts with a space instead of newlines for a more compact representation
                candidate_text += " ".join([text.replace("\n", " ") for text in contexts])
            else:
                # Original format with newlines
                candidate_text += "\n".join(contexts)
                
            all_contexts.append(candidate_text)
        
        return "\n\n".join(all_contexts)

    def get_top_candidates_with_llm(self,job_title: str, job_description: str, criteria : CriteriaSet,
                                 top_n: int =3, temperature: float = 0.2) -> LLMRankingResponse_1:
        """Search for top candidates using FAISS + LLM integration."""

        # Create job query
        job_query = f"Job Title: {job_title}\nJob Description: {job_description}"

        # First get top candidates using FAISS search
        search_k = top_n+3  # Get more candidates than needed for diversity
        top_candidates = self.get_top_candidates_with_faiss(job_query, search_k)
        print(f"Top candidate {top_candidates}")
        
        # Extract just the candidate names
        candidate_names = [name for name, score in top_candidates]

        # Get context chunks for these candidates
        candidate_contexts = self.get_candidate_contexts(candidate_names)
        context = self.format_candidate_contexts(candidate_contexts, compact=True)



        # # Create context string for LLM
        # all_contexts = []
        # for candidate, contexts in candidate_contexts.items():
        #     candidate_text = f"--- Candidate: {candidate} ---\n"
        #     candidate_text += "\n".join(contexts)
        #     all_contexts.append(candidate_text)

        # context = "\n\n".join(all_contexts)
        print(f"the context {context}")
        
        api_result=[]
        prompt_text = build_ranking_prompt(job_title, job_description, context)
        print(prompt_text)
        
        # Call LLM directly
        result = self.groq_llm.predict(prompt_text)
        print(result)
        # Extract and validate the result
        api_result = self._extract_json(result)
        api_result["faiss_candidates"]= candidate_names

        return api_result

    # def get_top_candidates_with_llm(self, job_title: str, job_description: str, criteria: CriteriaSet,
    #                             top_n: int = 3, temperature: float = 0.2) -> LLMRankingResponse_1:
    #     """Search for top candidates using FAISS + LLM integration."""

    #     # Create job query
    #     job_query = f"Job Title: {job_title}\nJob Description: {job_description}"

    #     # First get top candidates using FAISS search
    #     search_k = top_n + 3  # Get more candidates than needed for diversity
    #     top_candidates = self.get_top_candidates_with_faiss(job_query, search_k)
    #     print(f"Top candidate {top_candidates}")
        
    #     # Extract just the candidate names
    #     candidate_names = [name for name, score in top_candidates]

    #     # Get context chunks for these candidates
    #     candidate_contexts = self.get_candidate_contexts(candidate_names)

    #     # Create context string for LLM
    #     all_contexts = []
    #     for candidate, contexts in candidate_contexts.items():
    #         candidate_text = f"--- Candidate: {candidate} ---\n"
    #         candidate_text += "\n".join(contexts)
    #         all_contexts.append(candidate_text)

    #     context = "\n\n".join(all_contexts)
    #     print(f"the context {context}")
        
    #     api_result = []
    #     prompt_text = build_scoring_prompt(job_title, job_description, context, criteria)
        
        
    #     client = OpenAI(
    #         base_url="https://openrouter.ai/api/v1",
    #         api_key=Config.GROQ_API_KEY,  # Assuming you store the API key in the class
    #     )

    #     completion = client.chat.completions.create(
    #         # extra_headers={
    #         #     "HTTP-Referer": self.site_url,  # Optional, if you have these stored
    #         #     "X-Title": self.site_name,     # Optional
    #         # },
    #         model="deepseek/deepseek-r1:free",  # Or any other model you prefer
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt_text
    #             }
    #         ],
    #     )

    #     print(completion.choices[0].message.content)

        
    #     # Get the LLM response
    #     llm_response = completion.choices[0].message.content
    #     # print(llm_response)
        
    #     # Extract and validate the result
    #     api_result = self._extract_json(llm_response)
    #     api_result["faiss_candidates"] = candidate_names

    #     return api_result


    def _extract_json(self, text: str) -> dict:
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
            return raw_json
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from LLM response: {e}\nRaw string: {json_str}")

        # try:
        #     # Validate and convert to your API model
        #     # validated = LLMRankingResponse_1(**raw_json)
        # except ValidationError as ve:
        #     raise ValueError(f"LLM response does not match expected format: {ve}")

        # # Return as dict for API response
        # return validated.dict()