from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_community.chat_models import ChatOllama

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.memory.summary_buffer import (
    ConversationSummaryBufferMemory
)

from langchain_community.llms import Ollama

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
# MEMORY
# =========================================

memory_llm = Ollama(
    model="tinyllama"
)

memory = ConversationSummaryBufferMemory(
    llm=memory_llm,
    max_token_limit=300,
    return_messages=True
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
    chat_history: str

# =========================================
# FOLLOW-UP DETECTION
# =========================================

def is_followup_question(question):

    followup_words = [

        "it",
        "they",
        "them",
        "that",
        "those",
        "these",
        
        "available",

        "what about",
        "how about",
        "is it",
        "are they",
        "does it"
    ]

    q = question.lower()

    return any(word in q for word in followup_words)

# =========================================
# CLEAN TEXT
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

    best_doc = None

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

                            best_doc = content

    if best_score >= 2:

        return best_answer, best_doc, best_score

    return None, None, 0


# =========================================
# EVALUATION FUNCTION
# =========================================

def evaluate_rag_answer(answer, retrieved_docs):

    answer_words = set(
        clean_text(answer).split()
    )

    best_overlap = 0

    best_doc = None

    for doc in retrieved_docs:

        doc_words = set(
            clean_text(doc.page_content).split()
        )

        overlap = len(
            answer_words.intersection(doc_words)
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_doc = doc.page_content

    hallucination = False

    if best_overlap < 3:

        hallucination = True

    return {
        "hallucination": hallucination,
        "overlap_score": best_overlap,
        "source_document": best_doc
    }


# =========================================
# RAG NODE
# =========================================

def rag_node(state: ChatState):

    question = state["question"]

    # =====================================
    # LOAD MEMORY
    # =====================================

    chat_history = memory.load_memory_variables({})

    history = chat_history.get("history", [])

    history_text = "\n".join([

        str(msg.content)

        for msg in history
    ])

    print("\n========== MEMORY ==========")
    print(history_text)
    print("============================")

    # =====================================
    # ENHANCED QUERY
    # =====================================

    # =====================================
    # SMART MEMORY USAGE
    # =====================================

    if is_followup_question(question):

        enhanced_query = (
            history_text[-120:] + " " + question
        )

        print("\nUsing MEMORY for retrieval")

    else:

        enhanced_query = question

        print("\nUsing ONLY current question")

    print("\nEnhanced Query:")
    print(enhanced_query)

    # =====================================
    # VECTOR SEARCH
    # =====================================

    docs = db.similarity_search(
        enhanced_query,
        k=8
    )

    # =====================================
    # PRINT RETRIEVED DOCS
    # =====================================

    print("\n========== RETRIEVED DOCS ==========")

    for i, doc in enumerate(docs):

        print(f"\nDOC {i+1}:")
        print(doc.page_content[:300])

    print("\n====================================")

    # =====================================
    # DIRECT QA MATCH
    # =====================================

    direct_answer, source_doc, score = extract_direct_answer(
        enhanced_query,
        docs
    )

    if direct_answer:

        print("\n========== DIRECT QA MATCH ==========")
        print("Matched using direct retrieval")
        print("Similarity Score:", score)

        print("\nSOURCE DOCUMENT:")
        print(source_doc[:500])

        print("=====================================")

        memory.save_context(
            {"input": question},
            {"output": direct_answer}
        )

        evaluation_text = f"""

━━━━━━━━━━━━━━━
📊 RAG METRICS
━━━━━━━━━━━━━━━

✅ Grounded Response:
True

📄 Retrieval Score:
{score}

📚 Source:
{source_doc[:250]}

━━━━━━━━━━━━━━━
"""

        final_answer = (
            direct_answer +
            evaluation_text
        )

        return {
            "answer": final_answer
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
    # PROMPT
    # =====================================

    prompt = f"""
You are an SSN College assistant.

Use BOTH:
1. Conversation history
2. Retrieved context

to answer the latest question.

STRICT RULES:
- Keep answers under 2 sentences
- Answer directly
- Do not repeat full context
- Answer only from retrieved context
- If answer unavailable say:
I could not find that information.

Conversation History:
{history_text}

Retrieved Context:
{context}

Current Question:
{question}

Answer:
"""

    # =====================================
    # LLM RESPONSE
    # =====================================

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # =====================================
    # CLEAN OUTPUT
    # =====================================

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

            answer = "I could not find that information."

    answer = answer.split("\n")[0]

    if len(answer) > 300:

        answer = answer[:300]

    if not answer.strip():

        answer = "I could not find that information."

    # =====================================
    # EVALUATION LAYER
    # =====================================

    evaluation = evaluate_rag_answer(
        answer,
        docs
    )

    print("\n========== EVALUATION ==========")

    print("Hallucination Detected:",
          evaluation["hallucination"])

    print("Overlap Score:",
          evaluation["overlap_score"])

    print("\nMOST RELEVANT SOURCE DOCUMENT:\n")

    if evaluation["source_document"]:

        print(
            evaluation["source_document"][:700]
        )

    else:

        print("No supporting document found")

    print("\n================================")

    # =====================================
    # SAVE MEMORY
    # =====================================

    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    # =====================================
    # TELEGRAM OUTPUT
    # =====================================

    evaluation_text = f"""

━━━━━━━━━━━━━━━
📊 RAG METRICS
━━━━━━━━━━━━━━━

✅ Grounded Response:
{not evaluation["hallucination"]}

📄 Retrieval Score:
{evaluation["overlap_score"]}

📚 Source:
{evaluation["source_document"][:250] if evaluation["source_document"] else "No source found"}

━━━━━━━━━━━━━━━
"""

    final_answer = answer + evaluation_text

    return {
        "answer": final_answer
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
