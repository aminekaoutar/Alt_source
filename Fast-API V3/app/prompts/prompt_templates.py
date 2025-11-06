from langdetect import detect, LangDetectException
import json
from app.models.models import CriteriaSet

def build_dynamic_prompt(title: str, description: str) -> str:
    """Crée un prompt adapté à la langue de la description"""
    LANGUAGE_CONFIG = {
        'fr': {
            'role': "Expert en recrutement français",
            'output_example': """{
                "technical_skills": [{"name": "Python", "weight": 8}],
                "experience": [{"name": "5+ ans SaaS", "weight": 7}],
                "soft_skills": [{"name": "Gestion de projets", "weight": 6}]
            }"""
        },
        'en': {
            'role': "English Recruitment Expert",
            'output_example': """{
                "technical_skills": [{"name": "React", "weight": 9}],
                "experience": [{"name": "3+ years fintech", "weight": 8}],
                "soft_skills": [{"name": "Team leadership", "weight": 7}]
            }"""
        }
    }

    try:
        lang = detect(description) if description else 'en'
    except LangDetectException:
        lang = 'en'

    config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    return f"""
    [ROLE]
    {config['role']} - Analysez cette offre d'emploi pour identifier les critères vérifiables dans un CV.

    [TÂCHE]
    Titre : {title}
    Description : {description}

    [DIRECTIVES]
    1. Attribuer un poids de 1 (peu important) à 10 (critère clé)
    2. Ne pas inclure de champ 'reason'
    3. Format de sortie JSON valide

    [EXEMPLE]
    {config['output_example']}
    """

def build_jd_generation_prompt(job_query: str) -> str:
    """Crée un prompt pour générer une fiche de poste à partir d'une requête NLQ"""
    
    LANGUAGE_CONFIG = {
        'fr': {
            'role': "Expert en rédaction de fiches de poste français",
            'task_description': "Créez une fiche de poste détaillée et professionnelle",
            'output_example': """{
                "title": "Développeur Full Stack Senior",
                "summary": "Nous recherchons un développeur full stack expérimenté pour rejoindre notre équipe technique et contribuer au développement de nos applications web innovantes.",
                "responsibilities": [
                    "Développer et maintenir des applications web front-end et back-end",
                    "Collaborer avec l'équipe produit pour définir les spécifications techniques",
                    "Participer aux revues de code et maintenir la qualité du code"
                ],
                "requirements": [
                    "Diplôme en informatique ou expérience équivalente",
                    "Maîtrise de JavaScript, Python ou Java",
                    "Expérience avec les frameworks modernes (React, Vue.js, Django)"
                ],
                "experience_level": "Senior (5+ ans)",
                "technologies": ["JavaScript", "Python", "React", "PostgreSQL", "Docker"]
            }"""
        },
        'en': {
            'role': "Professional Job Description Writer",
            'task_description': "Create a detailed and professional job description",
            'output_example': """{
                "title": "Senior Full Stack Developer",
                "summary": "We are seeking an experienced full stack developer to join our technical team and contribute to the development of our innovative web applications.",
                "responsibilities": [
                    "Develop and maintain front-end and back-end web applications",
                    "Collaborate with product team to define technical specifications",
                    "Participate in code reviews and maintain code quality"
                ],
                "requirements": [
                    "Computer Science degree or equivalent experience",
                    "Proficiency in JavaScript, Python, or Java",
                    "Experience with modern frameworks (React, Vue.js, Django)"
                ],
                "experience_level": "Senior (5+ years)",
                "technologies": ["JavaScript", "Python", "React", "PostgreSQL", "Docker"]
            }"""
        }
    }

    try:
        lang = detect(job_query) if job_query else 'en'
    except LangDetectException:
        lang = 'en'

    config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    return f"""
    [ROLE]
    {config['role']} - Analysez cette requête pour créer une fiche de poste structurée et complète.

    [TÂCHE]
    Requête utilisateur : {job_query}

    [DIRECTIVES]
    1. Créer un titre de poste professionnel et accrocheur
    2. Rédiger un résumé concis mais informatif (2-3 phrases)
    3. Lister 4-6 responsabilités principales claires et spécifiques
    4. Définir 4-6 exigences essentielles (formation, compétences, expérience)
    5. Spécifier le niveau d'expérience requis (Junior, Intermédiaire, Senior, etc.)
    6. Identifier les technologies/outils clés (5-8 maximum)
    7. Utiliser un langage professionnel mais accessible
    8. Retourner uniquement du JSON valide sans commentaires

    [FORMAT DE SORTIE]
    {config['output_example']}

    [IMPORTANT]
    - Répondre UNIQUEMENT avec le JSON valide
    - Pas de texte avant ou après le JSON
    - Adapter le contenu au niveau de séniorité demandé
    - Être spécifique sur les technologies mentionnées
    """

def build_candidate_search_prompt(job_title: str, job_description: str, 
                                     context: str, top_n: int) -> str:
        """Creates a language-aware prompt for candidate search"""
        try:
            lang = detect(job_description) if job_description else 'en'
        except LangDetectException:
            lang = 'en'
            
        # Language-specific configurations
        LANGUAGE_CONFIG = {
            'fr': {
                'role': "Expert en recrutement français",
                'task': f"Analyser et classer les candidats pour le poste: {job_title}",
                'instructions': [
                    "Analyser chaque candidat en fonction des compétences et expériences requises",
                    f"Classer les {top_n} meilleurs candidats par ordre de pertinence",
                    "Pour chaque candidat, expliquer la correspondance et identifier les lacunes éventuelles",
                    "Identifier les compétences, qualifications ou expériences spécifiques qui correspondent aux exigences du poste"
                ],
                'output_format': """
                {
                    "top_candidates": [
                        {
                            "name": "Nom du candidat",
                            "match_score": 85,
                            "strengths": ["Force 1", "Force 2"],
                            "gaps": ["Lacune 1", "Lacune 2"],
                            "key_matches": ["Compétence 1 - Pertinence élevée", "Expérience 2 - Pertinence moyenne"]
                        }
                    ]
                }
                """
            },
            'en': {
                'role': "Expert Recruitment Consultant",
                'task': f"Analyze and rank candidates for the position: {job_title}",
                'instructions': [
                    "Analyze each candidate based on the required skills and experience",
                    f"Rank the top {top_n} candidates in order of relevance",
                    "For each candidate, explain their match and identify any potential gaps",
                    "Identify specific skills, qualifications, or experiences that match the job requirements"
                ],
                'output_format': """
                {
                    "top_candidates": [
                        {
                            "name": "Candidate name",
                            "match_score": 85,
                            "strengths": ["Strength 1", "Strength 2"],
                            "gaps": ["Gap 1", "Gap 2"],
                            "key_matches": ["Skill 1 - High relevance", "Experience 2 - Medium relevance"]
                        }
                    ]
                }
                """
            }
        }
        
        config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])
        
        instructions = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(config['instructions'])])
        
        return f"""
        [ROLE]
        {config['role']}
        
        [TASK]
        {config['task']}
        
        [JOB DESCRIPTION]
        {job_description}
        
        [CANDIDATE INFORMATION]
        {context}
        
        [INSTRUCTIONS]
        {instructions}
        
        [OUTPUT FORMAT]
        Return a valid JSON object with the following structure:
        {config['output_format']}
        
        Only return a valid JSON object, no additional text.
        """

def build_custom_query_prompt(custom_query: str, context: str, top_n: int) -> str:
        """Creates a language-aware prompt for custom candidate queries"""
        try:
            lang = detect(custom_query) if custom_query else 'en'
        except LangDetectException:
            lang = 'en'
            
        # Language-specific configurations
        LANGUAGE_CONFIG = {
            'fr': {
                'role': "Analyste expert de CV",
                'task': f"Analyser et classer les candidats selon la requête: {custom_query}",
                'instructions': [
                    f"Classer les {top_n} meilleurs candidats par ordre de pertinence pour cette requête",
                    "Pour chaque candidat, expliquer pourquoi ils correspondent aux critères de la requête",
                    "Inclure des qualifications, compétences ou expériences spécifiques de leur CV qui sont pertinentes",
                    "Inclure des qualifications, compétences ou expériences spécifiques de leur CV qui sont pertinentes",
                    "Inclure les informations de contact du candidat (téléphone, LinkedIn, email) si disponibles"

                ],
                'output_format': """
                {
                    "query": "La requête originale",
                    "candidates": [
                        {
                            "name": "Nom du candidat",
                            "relevance_score": 82,
                            "matching_points": ["Point correspondant 1 bien detaillé ", "Point correspondant 2 bien detaillé"],
                            "relevant_experience": "Description de l'expérience pertinente detaillé",
                            "contact": {
                                "phone": "Numéro de téléphone si disponible",
                                "linkedin": "URL du profil LinkedIn si disponible",
                                "email": "Adresse email si disponible"
                            }
                        }
                    ]
                }
                """


            },
            'en': {
                'role': "Expert CV Analyst",
                'task': f"Analyze and rank candidates according to the query: {custom_query}",
                'instructions': [
                    f"Rank the top {top_n} candidates in order of relevance for this query",
                    "For each candidate, explain why they match the query criteria",
                    "Include specific qualifications, skills, or experiences from their CV that are relevant",
                    "A justification of why this candidate matches the query, including a summary of key experiences or projects"
                    "Include candidate contact information (phone, LinkedIn, email) if available"

                ],
                'output_format': """
                {
                    "query": "The original query",
                    "candidates": [
                        {
                            "name": "Candidate name",
                            "relevance_score": 82,
                            "matching_points": ["Matching point 1 detaille detailed", "Matching point 2detailed"],
                            "relevant_experience": "Description of relevant experience detailed",
                            "contact": {
                                "phone": "Candidate phone number if available",
                                "linkedin": "LinkedIn profile URL if available",
                                "email": "Email address if available"
                            }
                        }
                    ]
                }
                """

            }
        }
        
        config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])
        
        instructions = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(config['instructions'])])
        
        return f"""
        [ROLE]
        {config['role']}
        
        [TASK]
        {config['task']}
        
        [CANDIDATE INFORMATION]
        {context}
        
        [INSTRUCTIONS]
        {instructions}
        
        [OUTPUT FORMAT]
        Return a valid JSON object with the following structure:
        {config['output_format']}
        
        Only return a valid JSON object, no additional text.
        """

# Add to your existing prompt template file
def build_ranking_prompt(job_title: str, job_description: str, context: str) -> str:
    """Build multilingual ranking prompt with structured JSON output"""
    LANGUAGE_CONFIG = {
        'fr': {
            'role': "Expert en recrutement français",
            'task': "Analysez les candidats et classez-les par ordre de pertinence",
            'directives': [
                "1. Fournir une liste classée des meilleurs candidats",
                "2. Inclure un score de correspondance (1-10) pour chaque candidat",
                "3. Lister les raisons de la correspondance et les lacunes",
                "4. Format JSON valide avec structure cohérente"
            ],
           'output_example': """{
                "ranked_candidates": [{
                    "candidate_name": "Extracted candidate name from context",
                    "match_score": 8.5,  
                    "reasons": ["5 ans d'expérience en Python"],
                    "gaps": ["Connaissance cloud limitée"],
                    "highlights": ["Leadership d'équipe"]
                }]
            }"""
        },
        'en': {
            'role': "English Recruitment Expert",
            'task': "Analyze candidates and rank them by suitability",
            'directives': [
                "1. Provide a ranked list of top candidates",
                "2. Include a match score (1-10) for each candidate",
                "3. List match reasons and potential gaps",
                "4. Valid JSON format with consistent structure"
            ],
            'output_example': """{
                "ranked_candidates": [{
                    "candidate_name": "John Doe",
                    "match_score": 9,
                    "reasons": ["5+ years Python experience", "Complex SaaS projects"],
                    "gaps": ["Limited cloud knowledge"],
                    "highlights": ["Team leadership", "Process optimization"]
                }]
            }"""
        }
    }

    try:
        lang = detect(job_description) if job_description else 'en'
    except LangDetectException:
        lang = 'en'

    config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    return f"""
    [ROLE]
    {config['role']} - {config['task']}

    [JOB DETAILS]
    Title: {job_title}
    Description: {job_description}

    [CANDIDATE CONTEXT]
    {context}

    [DIRECTIVES]
    {chr(10).join(config['directives'])}

    [OUTPUT EXAMPLE]
    {config['output_example']}
    """


def build_scoring_prompt(job_title: str, job_description: str, 
                       context: str, criteria: CriteriaSet, lang: str = 'en') -> str:
    """
    Build a prompt for ranking candidates against specific criteria
    
    Args:
        job_title: Target job position
        job_description: Detailed job description
        context: Formatted candidate contexts
        criteria: CriteriaSet object with evaluation criteria
        lang: Language code
    
    Returns:
        Formatted prompt for LLM ranking
    """
    # Format criteria for the prompt
    def format_criteria(criteria_list):
        return "\n".join([f"- {c.name} (weight: {c.weight})" for c in criteria_list])

    criteria_text = f"""
    Technical Skills:
    {format_criteria(criteria.technical_skills)}
    
    Experience Requirements:
    {format_criteria(criteria.experience)}
    
    Soft Skills:
    {format_criteria(criteria.soft_skills)}
    """

    template = {
        'en': f"""
        [ROLE]
        You are an expert HR recruiter evaluating multiple candidates for a position.
        
        [JOB DESCRIPTION]
        Position: {job_title}
        Details: {job_description}
        
        [EVALUATION CRITERIA]
        {criteria_text}
        
        [CANDIDATE PROFILES]
        {context}
        
        [TASK]
        1. Analyze all candidates against the evaluation criteria
        2. Rank top 3 candidates
        3. Provide scores for each criterion category
        4. Give detailed justifications for rankings
        
        [INSTRUCTIONS]
        - Compare candidates relatively
        - Consider criterion weights in scoring
        - Highlight key strengths/weaknesses
        - Be objective and use only provided information
        
        [REQUIRED OUTPUT FORMAT]
        ```json
        {{
          "ranking": [
            {{
              "candidate_name": "John Doe",
              "technical_score": 45,
              "experience_score": 30,
              "soft_skills_score": 25,
              "total_score": 100,
              "strengths": ["Python", "MLOps"],
              "weaknesses": ["Communication"],
              "justification": "Strong technical skills but needs communication training"
            }}
          ],
          "summary": "Overall candidate analysis summary..."
        }}
        ```
        """,
        'fr': f"""
        [RÔLE]
        Vous êtes un recruteur RH expert évaluant des candidats pour un poste.
        
        [DESCRIPTION DU POSTE]
        Poste: {job_title}
        Détails: {job_description}
        
        [CRITÈRES D'ÉVALUATION]
        {criteria_text}
        
        [PROFILS DES CANDIDATS]
        {context}
        
        [TÂCHE]
        1. Analyser tous les candidats selon les critères
        2. Classer les top 3 candidats
        3. Fournir des scores par catégorie
        4. Donner des justifications détaillées
        
        [INSTRUCTIONS]
        - Comparez les candidats entre eux
        - Tenez compte des poids des critères
        - Mettez en évidence forces/faiblesses
        - Soyez objectif et utilisez uniquement les informations fournies
        
        [FORMAT DE SORTIE REQUIS]
        ```json
        {{
          "ranking": [
            {{
              "candidate_name": "Jean Dupont",
              "technical_score": 45,
              "experience_score": 30,
              "soft_skills_score": 25,
              "total_score": 100,
              "strengths": ["Python", "MLOps"],
              "weaknesses": ["Communication"],
              "justification": "Compétences techniques solides mais besoin de formation en communication"
            }}
          ],
          "summary": "Résumé global de l'analyse..."
        }}
        ```
        """
    }
    
    try:
        detected_lang = detect(context) if context else lang
        return template.get(detected_lang, template['en'])
    except:
        return template['en']