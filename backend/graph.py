from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

import re


# =========================================
# LOAD LLM
# =========================================

llm = ChatOllama(
    model="tinyllama",
    temperature=0
)

# =========================================
# LOAD EMBEDDINGS
# =========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================
# LOAD VECTOR DB
# =========================================

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

# =========================================
# GRAPH STATE
# =========================================

class ChatState(TypedDict):

    question: str
    route: str
    answer: str


# =========================================
# CLEAN TEXT FUNCTION
# =========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.strip()


# =========================================
# GUARDRAIL NODE
# =========================================

def guardrail_node(state: ChatState):

    question = state["question"].lower()

    blocked_keywords = [

        "modi",
        "rahul gandhi",
        "politics",

        "password",
        "server access",
        "database password",

        "hack",
        "attack",
        "bypass",

        "confidential",
        "private data"
    ]

    for word in blocked_keywords:

        if word in question:

            return {
                "route": "blocked",
                "answer": (
                    "Sorry, I cannot answer that type of question."
                )
            }

    return {
        "route": "safe"
    }


# =========================================
# ROUTER NODE
# =========================================

def router_node(state: ChatState):

    question = state["question"].lower()

    escalation_keywords = [

        "hod",
        "principal",
        "dean",
        "complaint",
        "issue",
        "problem"
    ]

    if any(keyword in question for keyword in escalation_keywords):

        return {
            "route": "human"
        }

    return {
        "route": "rag"
    }


# =========================================
# DIRECT QA MATCH FUNCTION
# =========================================

def extract_direct_answer(question, docs):

    cleaned_question = clean_text(question)

    question_words = set(cleaned_question.split())

    best_answer = None

    best_score = 0

    for doc in docs:

        content = doc.page_content

        lines = content.split("\n")

        for i, line in enumerate(lines):

            line = line.strip()

            if line.lower().startswith("q:"):

                stored_question = (
                    line.replace("Q:", "")
                    .strip()
                )

                cleaned_stored = clean_text(stored_question)

                stored_words = set(cleaned_stored.split())

                common_words = (
                    question_words.intersection(stored_words)
                )

                score = len(common_words)

                # Find answer line
                if score > best_score:

                    if i + 1 < len(lines):

                        next_line = lines[i + 1].strip()

                        if next_line.lower().startswith("a:"):

                            answer = (
                                next_line.replace("A:", "")
                                .strip()
                            )

                            best_answer = answer

                            best_score = score

    # Minimum similarity threshold
    if best_score >= 2:

        return best_answer

    return None


# =========================================
# RAG NODE
# =========================================

def rag_node(state: ChatState):

    question = state["question"]

    # Retrieve docs
    docs = db.similarity_search(
        question,
        k=8
    )

    # =====================================
    # TRY DIRECT QA EXTRACTION FIRST
    # =====================================

    direct_answer = extract_direct_answer(
        question,
        docs
    )

    if direct_answer:

        return {
            "answer": direct_answer
        }

    # =====================================
    # BUILD CONTEXT
    # =====================================

    context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    if not context.strip():

        return {
            "answer": "I could not find that information."
        }

    # =====================================
    # LLM FALLBACK
    # =====================================

    prompt = f"""
You are an SSN College assistant.

STRICT RULES:
- Answer ONLY using the context
- Keep answer under 2 sentences
- Do not explain
- Do not add extra information
- If answer unavailable say exactly:
I could not find that information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # Remove junk
    banned_phrases = [

        "based on",
        "context",
        "rule",
        "question:",
        "answer:",
        "example:"
    ]

    for phrase in banned_phrases:

        if phrase.lower() in answer.lower():

            return {
                "answer": "I could not find that information."
            }

    answer = answer.split("\n")[0]

    if len(answer) > 300:
        answer = answer[:300]

    if not answer.strip():
        answer = "I could not find that information."

    return {
        "answer": answer
    }


# =========================================
# HUMAN NODE
# =========================================

def human_node(state: ChatState):

    return {
        "answer": (
            "Your query has been forwarded to the appropriate department staff."
        )
    }


# =========================================
# BLOCKED NODE
# =========================================

def blocked_node(state: ChatState):

    return {
        "answer": state["answer"]
    }


# =========================================
# ROUTE DECISION
# =========================================

def route_decision(state: ChatState):

    return state["route"]


# =========================================
# BUILD GRAPH
# =========================================

graph = StateGraph(ChatState)

graph.add_node("guardrail", guardrail_node)

graph.add_node("blocked", blocked_node)

graph.add_node("router", router_node)

graph.add_node("rag", rag_node)

graph.add_node("human", human_node)

graph.set_entry_point("guardrail")

graph.add_conditional_edges(
    "guardrail",
    route_decision,
    {
        "safe": "router",
        "blocked": "blocked"
    }
)

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "rag",
        "human": "human"
    }
)

graph.add_edge("blocked", END)

graph.add_edge("rag", END)

graph.add_edge("human", END)

app_graph = graph.compile()