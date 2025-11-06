import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Fast-API V3/app/db_tech_profile"

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    FAISS_INDEX_PATH = DATA_DIR / "tech_profile_vector_db.index"
    FAISS_METADATA_PATH = DATA_DIR / "tech_profile_vector_db_meta.pkl"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-8b-8192"
    TAVILY_API_KEY=os.getenv('TAVILY_API_KEY','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')
    GENT_SESSION_TIMEOUT: int = 3600  # 1 hour in seconds
    AGENT_MAX_SEARCH_RESULTS: int = 5
    AGENT_REQUIRE_APPROVAL_BY_DEFAULT: bool = True
  

# import os
# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# DATA_DIR = BASE_DIR / "Fast-API V3/app/db_tech_profile"

# class Config:
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
#     FAISS_INDEX_PATH = DATA_DIR / "tech_profile_vector_db.index"
#     FAISS_METADATA_PATH = DATA_DIR / "tech_profile_vector_db_meta.pkl"
#     EMBEDDING_MODEL = "all-MiniLM-L6-v2"
#     LLM_MODEL = "deepseek-chat"