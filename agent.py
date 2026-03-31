# agent.py: Implementation of AI Agent for the Conversation Module
# This module uses LangGraph (or LangChain Agent framework) to create an intelligent agent
# that handles multi-turn conversations, integrates RAG for knowledge retrieval,
# and provides features like progress tracking, error correction, and adaptive difficulty.
# It focuses on Spanish language learning scenarios, maintaining conversation state.
# Requires: pip install langgraph langchain langchain-openai
# Integrates with model.py (for LLM) and rag.py (for RAG).

import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rag import RAGSystem  # Import from rag.py (adjust if not class-based)
from model import client  # Import OpenAI client from model.py for consistency
import json

# API key and model (shared from model.py)
API_KEY = "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"
GENERATION_MODEL = "meta-llama/llama-3.1-8b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"

# Define conversation state (for multi-turn tracking)
class AgentState(TypedDict):
    messages: List[Dict[str, str]]  # History of messages: [{"role": "user/assistant", "content": "..."}]
    scenario: str  # Current conversation scenario (e.g., "restaurant")
    user_level: str  # User's proficiency level (e.g., "beginner")
    language: str  # Target language the user is learning (e.g., 'Spanish', 'French')
    progress: Dict[str, int]  # Progress tracking (e.g., {"words_mastered": 5, "accuracy": 80})
    last_correction: str  # Last error correction provided

# Initialize LLM for the agent
llm = ChatOpenAI(
    model=GENERATION_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.7,
)

# Define tools for the agent
@tool
def rag_retrieve(query: str, scenario: str = None, language: str = 'Spanish') -> str:
    """
    Tool to retrieve relevant knowledge using RAG.
    :param query: The query to search in the knowledge base.
    :param scenario: Optional scenario to filter (e.g., 'restaurant').
    :return: Retrieved context or response.
    """
    # Assuming rag.py has a RAGSystem class; instantiate if needed
    rag = RAGSystem()  # Or load from saved index
    # Pass language as part of filter metadata for RAG; RAGSystem.query will use it if present
    filt = {"scenario": scenario} if scenario else {}
    if language:
        filt["language"] = language
    result = rag.query(query, filter=filt if filt else None)
    return result

@tool
def update_progress(metric: str, value: int) -> str:
    """
    Tool to update user's learning progress.
    :param metric: The metric to update (e.g., 'words_mastered', 'accuracy').
    :param value: The new value or increment.
    :return: Confirmation message.
    """
    # In a real app, persist to localStorage or DB; here, simulate
    return f"Updated {metric} to {value}."

@tool
def generate_exercise(level: str, topic: str) -> str:
    """
    Tool to generate a language exercise.
    :param level: User's level (e.g., 'beginner').
    :param topic: Topic (e.g., 'grammar', 'vocabulary').
    :return: Generated exercise as string.
    """
    # Simple generation; in full, use LLM
    return f"Exercise for {level} level on {topic}: Translate 'Hello, how are you?' to Spanish."

# Bind tools to LLM
tools = [rag_retrieve, update_progress, generate_exercise]
llm_with_tools = llm.bind_tools(tools)

# System prompt for the agent (educational, conversational)
# Escaped curly braces to prevent KeyError in formatting
system_prompt = """
You are a {language} language learning assistant in a conversation simulation. Role-play strictly as the following based on the current scenario, but reference previous conversation history if relevant:
- If scenario is 'restaurant', you are a waiter/waitress. Handle reservations, ordering, menu recommendations.
- If scenario is 'hotel', you are a hotel receptionist. Handle check-in, room requests, services.
- If scenario is 'shopping', you are a shop assistant. Help with items, sizes, prices.
- If scenario is 'doctor', you are a doctor. Discuss symptoms, advice.
- If scenario is 'interview', you are a job interviewer. Ask about experience, skills.
Always adjust to the current scenario: {scenario}, even if history is from a different scenario. Use the user's level {user_level} and the user's target language {language} to adjust complexity.
Use tools when needed: retrieve knowledge with RAG for accurate responses, update progress based on user performance, or generate exercises for practice.
Maintain conversation flow across scenarios if possible, correct errors politely, and adapt to user's level.
Keep responses engaging: Respond ONLY in JSON format: {{ "response": "{language} response here", "translation": "English translation here", "tips": "1-2 tips or corrections (optional)" }}.
Do not add extra notes or brackets in the response field.
If user input is invalid or nonsense (e.g., random characters), respond politely like "Lo siento, no entendí eso. ¿Puedes repetirlo?" and translation "Sorry, I didn't understand that. Can you repeat?" and suggest clarification.
Suggest increasing difficulty if progress is good.
"""

# Clarify feedback focus by user level so the LLM produces the expected type of feedback
system_prompt = system_prompt + "\n" + (
    "Level guidance:\n"
    "- beginner: Prioritize vocabulary support and simple, bite-sized corrections. Focus feedback on correct word choice, provide 1-2 replacement words or quick synonyms, and keep grammar comments minimal and concise. Keep sentences short and clear. Use very simple phrasing in explanations.\n"
    "- intermediate: Prioritize grammar and sentence-structure corrections. Highlight common grammatical errors, explain why they occurred briefly, and provide corrected example sentences. Include 1-2 tips to improve grammar.\n"
    "- advanced: Prioritize overall fluency, idiomatic expressions, and register. Offer suggestions for more natural phrasing, stylistic alternatives, and advanced collocations. Keep corrections concise but give 1-2 high-impact fluency tips.\n"
    "Language requirement for feedback:\n"
    "- Always produce the `tips` and any corrective feedback in clear English, regardless of the user's input language.\n"
    "- JSON-only: The assistant MUST return strictly valid JSON with exactly the keys `response`, `translation`, and `tips` (tips may be empty). Do NOT include any extra text before or after the JSON.\n"
    "Language-specific formatting rules:\n"
    "- For Mandarin/Chinese (language value 'Mandarin' or 'Chinese'):\n"
    "  * `response`: Use Simplified Chinese characters (UTF-8), do NOT use pinyin in the `response` field.\n"
    "  * `translation`: Provide a fluent English translation of the `response`.\n"
    "  * `tips`: Provide corrective feedback in clear English. When giving example phrases or pronunciation help, include pinyin transliteration in parentheses after the Chinese text, e.g., '你好 (nǐ hǎo)'. Use numbered tones or diacritics consistently.\n"
    "- For Japanese: `response` should be in natural Japanese (kanji/kana), `translation` in English, `tips` in English.\n"
    "- For all other languages: the `response` should be in the target language script, `translation` in English, and `tips` in English.\n"
    "When producing the JSON response, place the primary learning feedback into the `tips` field (in English) and make the `response` a natural reply in the user's target language ({language}) at the user's level. The `translation` field should be the English translation of that reply."
)

# Define agent nodes
def agent_node(state: AgentState) -> AgentState:
    """
    Main agent node: Decide actions based on state and user input.
    """
    # Ensure message roles are normalized to what the LLM expects.
    # The frontend or previous code may use 'ai' for assistant; map that to 'assistant'.
    def normalize_messages(msgs):
        normalized = []
        for m in msgs:
            role = m.get("role", "user")
            # normalize common variants
            if role == 'ai':
                role = 'assistant'
            # Do not pass non-standard roles like 'tool' to the LLM; treat them as 'system'
            if role == 'tool':
                role = 'system'
            if role not in ('system', 'user', 'assistant'):
                # Treat unknown roles as user to avoid confusing the LLM
                role = 'user'
            normalized.append({"role": role, "content": m.get("content", "")})
        return normalized

    messages = normalize_messages(state["messages"])
    # Provide the language into the system prompt so the model knows which language to role-play
    system_formatted = system_prompt.format(user_level=state["user_level"], scenario=state["scenario"], language=state.get("language", "Spanish"))
    # Build messages for the LLM using normalized roles.
    messages_for_llm = [{"role": "system", "content": system_formatted}] + messages
    
    try:
        response = llm_with_tools.invoke(messages_for_llm)
    except Exception as e:
        print(f"LLM invoke error: {str(e)}")
        response = type('MockResponse', (), {"tool_calls": [], "content": '{"response": "Lo siento, hubo un error.", "translation": "Sorry, there was an error.", "tips": ""}'})()
    
    # If tool calls, execute them and append as tool messages so the final LLM call can see the outputs.
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for call in response.tool_calls:
            tool_name = call.get("name") or call.get("tool", "unknown_tool")
            args = call.get("args", {}) if isinstance(call, dict) else {}
            result = "Tool not found."
            try:
                if tool_name == "rag_retrieve":
                    result = rag_retrieve(args.get('query') if isinstance(args, dict) else args)
                elif tool_name == "update_progress":
                    # args may be a dict like {'metric':..., 'value':...}
                    if isinstance(args, dict):
                        result = update_progress(args.get('metric', ''), args.get('value', 0))
                    else:
                        result = update_progress('unknown', 0)
                elif tool_name == "generate_exercise":
                    if isinstance(args, dict):
                        result = generate_exercise(args.get('level', 'beginner'), args.get('topic', 'general'))
                    else:
                        result = generate_exercise('beginner', 'general')
            except Exception as te:
                result = f"Tool execution error: {str(te)}"
            # Append tool result as a 'system' role message so the LLM receives the info
            messages.append({"role": "system", "content": f"[tool:{tool_name}] {result}"})
    
    # Generate final response
    # Final LLM call: provide system prompt and the normalized message list (including any tool outputs).
    final_messages = [{"role": "system", "content": system_formatted}] + messages
    try:
        final_response = llm.invoke(final_messages)
    except Exception as e:
        print(f"LLM final invoke error: {str(e)}")
        final_response = type('MockResponse', (), {"content": '{"response": "Lo siento, hubo un error. ¿Puedes intentarlo de nuevo?", "translation": "Sorry, there was an error. Can you try again?", "tips": ""}'})()

    content = final_response.content if hasattr(final_response, 'content') else str(final_response)
    # Append assistant response to the stored state with normalized role
    state_msgs = state.get("messages", [])
    state_msgs.append({"role": "assistant", "content": content})
    state["messages"] = state_msgs
    
    # Update state (e.g., simulate progress)
    if "correct" in content.lower():  # Simple heuristic
        state["progress"]["accuracy"] = min(100, state["progress"].get("accuracy", 0) + 10)
    
    return state

# Build the graph
graph = StateGraph(state_schema=AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
app_graph = graph.compile()  # Renamed to avoid conflict

# Function to run the agent
def run_agent(user_input: str, state: AgentState = None) -> Dict[str, Any]:
    """
    Run the agent with user input and optional initial state.
    :param user_input: User's message.
    :param state: Existing state (for multi-turn).
    :return: Updated state with response.
    """
    if state is None:
        state = AgentState(
            messages=[],
            scenario="restaurant",  # Default
            user_level="beginner",   # Default
            language="Spanish",
            progress={"words_mastered": 0, "accuracy": 0},
            last_correction=""
        )
    
    state["messages"].append({"role": "user", "content": user_input})
    result = app_graph.invoke(state)
    return result

# Example usage (for testing; integrate with main.py)
if __name__ == "__main__":
    initial_state = None
    response1 = run_agent("Hola, ¿cómo estás?")
    print("Agent Response:", response1["messages"][-1]["content"])
    
    # Multi-turn
    response2 = run_agent("¿Tienes una mesa para dos?", state=response1)
    print("Agent Response:", response2["messages"][-1]["content"])