from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


# =========================================
# LOAD LOCAL LLM
# =========================================

llm = ChatOllama(
    model="tinyllama"
)


# =========================================
# LOAD EMBEDDING MODEL
# =========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================
# LOAD VECTOR DATABASE
# =========================================

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)


# =========================================
# DEFINE GRAPH STATE
# =========================================

class ChatState(TypedDict):

    question: str
    route: str
    answer: str


# =========================================
# ROUTER NODE
# =========================================
# =========================================
# ROUTER NODE
# =========================================
# =========================================
# GUARDRAIL NODE
# =========================================

def guardrail_node(state: ChatState):

    question = state["question"].lower()

    # Blocked topics
    blocked_keywords = [

        # Politics
        "modi",
        "rahul gandhi",
        "politics",
        "political party",

        # Security
        "password",
        "admin access",
        "server access",
        "database password",
        "security system",

        # Harmful
        "hack",
        "bypass",
        "attack",

        # Sensitive
        "confidential",
        "private data"
    ]

    # Check blocked content
    for word in blocked_keywords:

        if word in question:

            return {
                "route": "blocked",
                "answer": (
                    "Sorry, I cannot answer "
                    "that type of question."
                )
            }

    # Safe query
    return {
        "route": "safe"
    }

def router_node(state: ChatState):

    question = state["question"].lower()

    # Human escalation keywords
    escalation_keywords = [
        "hod",
        "principal",
        "dean",
        "staff",
        "faculty",
        "admission office",
        "complaint",
        "issue",
        "problem"
    ]

    # Escalation phrases
    escalation_phrases = [
        "talk to",
        "speak to",
        "contact"
    ]

    # Check escalation keywords
    keyword_match = any(
        keyword in question
        for keyword in escalation_keywords
    )

    # Check escalation phrase + person/entity
    phrase_match = False

    for phrase in escalation_phrases:

        if phrase in question:

            if (
                "hod" in question
                or "principal" in question
                or "staff" in question
                or "faculty" in question
                or "office" in question
                or "teacher" in question
            ):

                phrase_match = True

    # Route to human only if meaningful escalation
    if keyword_match or phrase_match:

        return {
            "route": "human"
        }

    # Otherwise use RAG
    return {
        "route": "rag"
    }


# =========================================
# RAG NODE
# =========================================

def rag_node(state: ChatState):

    question = state["question"]

    # Retrieve documents
    docs = db.similarity_search(
        question,
        k=3
    )

    # Build context
    context_parts = []

    for doc in docs:

        content = doc.page_content.strip()

        if len(content) > 20:
            context_parts.append(content)

    context = "\n".join(context_parts)

    # If no context found
    if not context.strip():

        return {
            "answer": "I could not find that information."
        }

    # STRICT PROMPT
    prompt = f"""
You are an SSN College information assistant.

Answer ONLY using the information provided below.

IMPORTANT RULES:
- Return ONLY the final answer.
- Maximum 1 sentence.
- DO NOT explain.
- DO NOT repeat the question.
- DO NOT say "based on the context".
- DO NOT mention rules.
- DO NOT generate examples.
- If answer is unavailable, reply EXACTLY:
I could not find that information.

INFORMATION:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # Cleanup
    unwanted_phrases = [
        "based on",
        "context",
        "rule",
        "question:",
        "answer:",
        "example:"
    ]

    for phrase in unwanted_phrases:

        if phrase.lower() in answer.lower():

            return {
                "answer": "I could not find that information."
            }

    # Limit output size
    answer = answer.split("\n")[0]

    if len(answer) > 250:
        answer = answer[:250]

    return {
        "answer": answer
    }

# =========================================
# HUMAN ESCALATION NODE
# =========================================

def human_node(state: ChatState):

    return {
        "answer": (
            "Your query has been forwarded "
            "to the appropriate department staff."
        )
    }

# =========================================
# BLOCK NODE
# =========================================

def blocked_node(state: ChatState):

    return {
        "answer": state["answer"]
    }

# =========================================
# ROUTING DECISION
# =========================================

def route_decision(state: ChatState):

    return state["route"]


# =========================================
# BUILD LANGGRAPH
# =========================================

graph = StateGraph(ChatState)

# Add nodes
graph.add_node("guardrail", guardrail_node)
graph.add_node("blocked", blocked_node)
graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("human", human_node)

# Entry point
graph.set_entry_point("guardrail")

# Guardrail routing
graph.add_conditional_edges(
    "guardrail",
    route_decision,
    {
        "safe": "router",
        "blocked": "blocked"
    }
)

# Conditional routing
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "rag",
        "human": "human"
    }
)

# End nodes
graph.add_edge("blocked", END)
graph.add_edge("rag", END)
graph.add_edge("human", END)

# Compile graph
app_graph = graph.compile()