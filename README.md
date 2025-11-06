# 🧠 AltSource.AI – Intelligent Candidate Preselection Platform

## 📘 Overview
**AltSource.AI** is an intelligent recruitment platform designed to assist HR teams in the **preselection and analysis of candidates** using **Large Language Models (LLMs)** and **semantic vector search**.  
It enhances traditional Applicant Tracking Systems (ATS) by introducing AI-driven analysis, contextual matching, and candidate profile enrichment.

this solution demonstrates how modern AI tools can transform digital recruitment workflows.

---

## 🎯 Objectives

- Automate **semantic matching** between job descriptions and CVs  
- Maintain **up-to-date and enriched candidate profiles** using AI  
- Provide **transparent and explainable scoring** for recruiters  
- Offer a **modern HR interface** inspired by SaaS recruitment tools  
- Integrate seamlessly with existing **ATS platforms** (e.g., BoondManager)

---

## 🏗️ System Architecture

The solution is built on a **modular and scalable architecture** combining vector databases, APIs, and LLM orchestration.

```
Frontend (Next.js + TypeScript)
        │
        ▼
FastAPI Backend  ──► FAISS Vector Store
        │
        ├── LLM Engine (LangChain + LangGraph)
        ├── Database (MySQL / SQLite)
        └── ATS Integrations (via REST API)
```

### 🧠 Core Components
| Component | Technology | Description |
|------------|-------------|-------------|
| **Frontend (UI)** | Next.js + TypeScript | Dynamic HR dashboard, campaign management, candidate visualization |
| **Backend API** | FastAPI | Exposes endpoints for matching, scoring, and ATS integration |
| **LLM Layer** | LangChain + LangGraph | Semantic reasoning, JD optimization, conversational assistance |
| **Vector Search** | FAISS | Fast semantic search and ranking of candidate profiles |
| **Database** | MySQL / SQLite | Storage of candidate data, job descriptions, and analytics |
| **AI Models** | Sentence-BERT / OpenAI | Embedding generation and scoring |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/altsource-ai.git
cd altsource-ai
```

### 2️⃣ Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3️⃣ Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4️⃣ Environment Variables
Create a `.env` file in both `frontend/` and `backend/` directories and configure:
```
OPENAI_API_KEY=your_api_key
DATABASE_URL=mysql://user:password@localhost/altsource
```

---

## 💻 Main Interfaces (UI Screens)

### 🔐 1. Login Page
Simple authentication interface for recruiters.  
![Login Page](assets/login.png)

### 📊 2. Dashboard
Centralized view of all active and archived recruitment campaigns.  
![Dashboard](assets/dashboard.png)

### 🧾 3. CV Base & ATS Integration
Upload CV datasets (CSV) and synchronize candidate data with ATS platforms.  
![CV Base](assets/cv-base.png)

### 📂 4. Campaign Management
Create and monitor sourcing campaigns — view active and archived searches.  
![Campaign Management](assets/campaigns.png)

### ✍️ 5. Job Description Optimization
AI-powered assistant (LangGraph Agent) to refine and optimize job descriptions.  
![JD Optimization](assets/jd-optimization.png)

### 🔍 6. Candidate Matching
Displays ranked candidates based on semantic similarity scores between JD and CVs.  
![Candidate Matching](assets/matching.png)

### 👤 7. Candidate Detail View
Shows full profile data, extracted skills, experience, and AI-generated matching explanations.  
![Candidate Detail](assets/candidate-detail.png)

---

## 🧪 Example Workflow

1. Recruiter creates a **new sourcing campaign**.  
2. The JD is **vectorized and matched** against existing candidate embeddings.  
3. The system generates a **semantic ranking** of best-fit candidates.  
4. Recruiter can **optimize JD**, **contact candidates**, and **track campaign analytics**.

---

## 📊 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| **Frontend** | Next.js, TypeScript, TailwindCSS |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy |
| **AI / NLP** | HuggingFace Transformers, Sentence-BERT, LangChain, LangGraph |
| **Database** | MySQL, SQLite |
| **Vector Store** | FAISS |
| **Integration & Tools** | Postman, Swagger, Docker (optional) |



## 🚀 Future Improvements

- Integration with additional ATS platforms (Workable, Greenhouse)
- Implementation of on-premise open-source LLMs
- Advanced analytics dashboards
- Real-time collaboration and chat features
- GDPR-compliant anonymization module

---

## 👤 Author

**Amine Kaoutar**  
📧 [[LinkedIn Profile](https://www.linkedin.com/in/amine-kaoutar/)] 

**Abdelhakim Elhaiba**  
📧 [[LinkedIn Profile](https://www.linkedin.com/in/hakimelhaiba/)]  

---

## 🪪 License
This project is licensed under the **MIT License** – feel free to use and adapt it for research and educational purposes.

---

⭐ *If you find this project useful, don’t forget to give it a star on GitHub!*
