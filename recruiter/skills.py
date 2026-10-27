"""Skill taxonomy and normalisation.

The taxonomy maps canonical skill names to a category and a list of aliases.
It is used by the rule-based extractors and also to normalise whatever an LLM
returns, so "py", "Python3" and "python" all collapse to "Python".
"""

from __future__ import annotations

import re
from functools import lru_cache

# canonical name -> (category, aliases)
TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # Programming languages
    "Python": ("language", ["python3", "python 3"]),
    "Java": ("language", []),
    "JavaScript": ("language", ["js", "ecmascript", "es6"]),
    "TypeScript": ("language", ["ts"]),
    "C++": ("language", ["cpp"]),
    "C#": ("language", ["c sharp", "csharp"]),
    "Go": ("language", ["golang"]),
    "Rust": ("language", []),
    "R": ("language", ["r programming", "rstudio"]),
    "Scala": ("language", []),
    "Kotlin": ("language", []),
    "SQL": ("language", ["t-sql", "pl/sql", "plsql", "ansi sql"]),
    "Bash": ("language", ["shell scripting", "shell script", "powershell"]),
    "MATLAB": ("language", []),
    # ML / AI
    "Machine Learning": ("ai_ml", ["ml", "machine-learning", "predictive modelling", "predictive modeling"]),
    "Deep Learning": ("ai_ml", ["deep-learning", "neural networks", "neural network"]),
    "NLP": ("ai_ml", ["natural language processing", "text mining", "text analytics"]),
    "Computer Vision": ("ai_ml", ["image recognition", "object detection"]),
    "LLMs": ("ai_ml", ["llm", "large language models", "large language model", "generative ai", "genai", "gen ai"]),
    "Prompt Engineering": ("ai_ml", ["prompt design", "prompting"]),
    "RAG": ("ai_ml", ["retrieval augmented generation", "retrieval-augmented generation"]),
    "AI Agents": ("ai_ml", ["agentic ai", "multi-agent", "multi agent", "llm agents", "agent workflows", "agentic workflows"]),
    "LangChain": ("ai_ml", []),
    "LlamaIndex": ("ai_ml", ["llama index", "llama-index"]),
    "Hugging Face": ("ai_ml", ["huggingface", "transformers library", "hugging face transformers"]),
    "PyTorch": ("ai_ml", ["torch"]),
    "TensorFlow": ("ai_ml", ["tf", "keras"]),
    "scikit-learn": ("ai_ml", ["sklearn", "scikit learn"]),
    "XGBoost": ("ai_ml", ["lightgbm", "gradient boosting"]),
    "MLOps": ("ai_ml", ["ml ops", "model deployment", "model monitoring"]),
    "MLflow": ("ai_ml", []),
    "Model Evaluation": ("ai_ml", ["evaluation metrics", "model validation", "a/b testing", "ab testing"]),
    "Vector Databases": ("ai_ml", ["vector database", "vector db", "pinecone", "faiss", "chromadb", "chroma", "weaviate", "qdrant", "pgvector"]),
    "Ollama": ("ai_ml", []),
    "OpenAI API": ("ai_ml", ["openai", "gpt-4", "gpt-3.5", "chatgpt api"]),
    "Statistics": ("data", ["statistical analysis", "hypothesis testing", "regression analysis", "statistical modelling", "statistical modeling"]),
    # Data
    "Pandas": ("data", []),
    "NumPy": ("data", ["numpy"]),
    "Data Analysis": ("data", ["data analytics", "exploratory data analysis", "eda"]),
    "Data Visualization": ("data", ["data visualisation", "matplotlib", "seaborn", "plotly"]),
    "Power BI": ("data", ["powerbi"]),
    "Tableau": ("data", []),
    "Excel": ("data", ["microsoft excel", "ms excel", "advanced excel", "vlookup", "pivot tables"]),
    "Spark": ("data", ["apache spark", "pyspark"]),
    "Hadoop": ("data", []),
    "Airflow": ("data", ["apache airflow"]),
    "ETL": ("data", ["data pipelines", "data pipeline", "elt"]),
    "Data Warehousing": ("data", ["data warehouse", "snowflake", "bigquery", "redshift"]),
    "dbt": ("data", []),
    "PostgreSQL": ("database", ["postgres"]),
    "MySQL": ("database", []),
    "SQLite": ("database", []),
    "MongoDB": ("database", ["mongo"]),
    "Redis": ("database", []),
    # Web / backend
    "Streamlit": ("web", []),
    "FastAPI": ("web", ["fast api"]),
    "Flask": ("web", []),
    "Django": ("web", []),
    "REST APIs": ("web", ["rest api", "restful", "restful apis", "api development", "rest"]),
    "GraphQL": ("web", []),
    "React": ("web", ["react.js", "reactjs"]),
    "Node.js": ("web", ["node", "nodejs", "express.js", "express"]),
    "HTML/CSS": ("web", ["html", "css", "html5", "css3"]),
    "Spring Boot": ("web", ["spring"]),
    "Microservices": ("web", ["microservice", "service-oriented architecture"]),
    # Cloud / DevOps
    "AWS": ("cloud", ["amazon web services", "ec2", "s3", "lambda", "sagemaker"]),
    "Azure": ("cloud", ["microsoft azure", "azure ml", "azure openai"]),
    "GCP": ("cloud", ["google cloud", "google cloud platform", "vertex ai"]),
    "Docker": ("devops", ["containerisation", "containerization", "containers"]),
    "Kubernetes": ("devops", ["k8s"]),
    "CI/CD": ("devops", ["ci cd", "continuous integration", "github actions", "jenkins", "gitlab ci"]),
    "Git": ("devops", ["github", "gitlab", "version control"]),
    "Terraform": ("devops", ["infrastructure as code", "iac"]),
    "Linux": ("devops", ["unix", "ubuntu"]),
    # Testing / quality
    "Software Testing": ("quality", ["qa", "quality assurance", "test automation", "automated testing", "unit testing", "integration testing"]),
    "Pytest": ("quality", ["py.test"]),
    "Selenium": ("quality", []),
    # Practices / soft
    "Agile": ("practice", ["scrum", "kanban", "sprint planning"]),
    "System Design": ("practice", ["solution architecture", "software architecture", "architecture design"]),
    "Communication": ("soft", ["stakeholder communication", "presentation skills", "written communication"]),
    "Stakeholder Management": ("soft", ["stakeholder engagement", "client management", "client-facing"]),
    "Leadership": ("soft", ["team lead", "mentoring", "mentored", "people management"]),
    "Project Management": ("practice", ["pmp", "programme management", "program management"]),
    "Problem Solving": ("soft", ["analytical thinking", "critical thinking"]),
    # Business / recruitment domain
    "Recruitment": ("business", ["talent acquisition", "sourcing", "recruiting"]),
    "Financial Analysis": ("business", ["financial modelling", "financial modeling", "forecasting"]),
    "Digital Marketing": ("business", ["seo", "sem", "google analytics", "social media marketing"]),
    "Product Management": ("business", ["product roadmap", "product owner"]),
}

# Ambiguous short aliases only count when written in a "tech" context; we keep a
# list of aliases that need an exact case-sensitive match to avoid false hits
# ("go" the verb, "r" the letter, "rest" the noun, "tf"...).
CASE_SENSITIVE = {"Go", "R", "TS", "TF", "REST", "ML", "JS", "QA", "SEO", "SEM", "EDA", "ELT", "IaC", "S3", "EC2"}
AMBIGUOUS_ALIASES = {"go", "r", "rest", "tf", "ts", "node", "express", "spring", "torch", "containers", "lambda", "chroma", "sourcing", "forecasting", "s3", "ec2"}

CERTIFICATION_PATTERNS = [
    r"AWS Certified[\w \-]*",
    r"Azure (?:AI|Data|Fundamentals|Administrator|Solutions)[\w \-]*(?:\(\w+-\d+\))?",
    r"Google (?:Cloud )?(?:Professional|Associate)[\w \-]*",
    r"TensorFlow Developer Certificate",
    r"PMP",
    r"Certified Scrum Master|CSM",
    r"CKA|Certified Kubernetes Administrator",
    r"ISTQB[\w \-]*",
]

DEGREE_PATTERNS = [
    (r"\b(ph\.?d|doctorate)\b", "PhD"),
    (r"\b(m\.?sc|master'?s?|m\.?s\.|mba|m\.?eng|m\.?tech)\b", "Master's"),
    (r"\b(b\.?sc|bachelor'?s?|b\.?s\.|b\.?a\.|b\.?eng|b\.?tech|undergraduate degree)\b", "Bachelor's"),
    (r"\b(diploma|associate degree)\b", "Diploma"),
]


def _alias_pattern(alias: str, case_sensitive: bool) -> re.Pattern[str]:
    # Custom boundaries so "C++", "C#", ".NET", "Node.js" and "CI/CD" match cleanly.
    escaped = re.escape(alias)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![\w+#.-]){escaped}(?![\w+#&]|\.\w)", flags)


@lru_cache(maxsize=1)
def compiled_taxonomy() -> list[tuple[str, str, list[re.Pattern[str]]]]:
    compiled = []
    for canonical, (category, aliases) in TAXONOMY.items():
        patterns = []
        for alias in [canonical, *aliases]:
            if alias.lower() in AMBIGUOUS_ALIASES and alias != canonical:
                # Ambiguous aliases ("rest", "node", "spring") are skipped for free
                # text; they are still honoured by normalise_skill().
                continue
            case_sensitive = alias in CASE_SENSITIVE or len(alias) <= 2
            patterns.append(_alias_pattern(alias, case_sensitive=case_sensitive))
        compiled.append((canonical, category, patterns))
    return compiled


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, (_cat, aliases) in TAXONOMY.items():
        index[_norm(canonical)] = canonical
        for alias in aliases:
            index.setdefault(_norm(alias), canonical)
    return index


def _norm(text: str) -> str:
    return re.sub(r"[\s_]+", " ", text.strip().lower())


def normalise_skill(name: str) -> str:
    """Map any spelling of a skill to its canonical name (or tidy it if unknown)."""
    key = _norm(name)
    if key in _alias_index():
        return _alias_index()[key]
    # Try stripping version numbers / trailing qualifiers: "Python 3.11", "React 18"
    stripped = re.sub(r"\s*v?\d+(\.\d+)*$", "", key)
    if stripped in _alias_index():
        return _alias_index()[stripped]
    return name.strip()


def category_of(skill: str) -> str:
    canonical = normalise_skill(skill)
    return TAXONOMY.get(canonical, ("general", []))[0]


def find_skills(text: str) -> dict[str, list[tuple[int, int]]]:
    """Return {canonical_skill: [(start, end), ...]} for every taxonomy hit in text."""
    hits: dict[str, list[tuple[int, int]]] = {}
    for canonical, _category, patterns in compiled_taxonomy():
        spans = []
        for pattern in patterns:
            spans.extend(m.span() for m in pattern.finditer(text))
        if spans:
            hits[canonical] = sorted(set(spans))
    return hits


def find_certifications(text: str) -> list[str]:
    found: list[str] = []
    for pattern in CERTIFICATION_PATTERNS:
        for m in re.finditer(pattern, text):
            cert = m.group(0).strip(" -")
            if cert and cert not in found:
                found.append(cert)
    return found


def find_degrees(text: str) -> list[str]:
    found = []
    for pattern, label in DEGREE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)
    return found


DEGREE_RANK = {"Diploma": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}


# Transferable-skill groups: having one member earns partial credit for another.
RELATED_GROUPS: list[set[str]] = [
    {"AWS", "Azure", "GCP"},
    {"PyTorch", "TensorFlow"},
    {"Flask", "FastAPI", "Django"},
    {"PostgreSQL", "MySQL", "SQLite"},
    {"Power BI", "Tableau"},
    {"LangChain", "LlamaIndex"},
    {"Docker", "Kubernetes"},
    {"Java", "Kotlin", "Scala", "C#"},
    {"JavaScript", "TypeScript"},
    {"Spark", "Hadoop"},
    {"Airflow", "ETL", "dbt"},
    {"OpenAI API", "Ollama", "LLMs"},
    {"Pytest", "Software Testing", "Selenium"},
    {"Machine Learning", "scikit-learn", "XGBoost"},
    {"RAG", "Vector Databases", "LlamaIndex"},
]


def related_skills(skill: str) -> set[str]:
    out: set[str] = set()
    for group in RELATED_GROUPS:
        if skill in group:
            out |= group - {skill}
    return out


# Skills whose demonstrated use implies another skill (building a FastAPI service is REST API work).
IMPLIED_SKILLS: dict[str, str] = {
    "FastAPI": "REST APIs",
    "Flask": "REST APIs",
    "Django": "REST APIs",
    "Spring Boot": "REST APIs",
    "Pytest": "Software Testing",
    "Selenium": "Software Testing",
}
