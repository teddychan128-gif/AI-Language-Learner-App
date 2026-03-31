# main.py: Integration of model.py, rag.py, and agent.py for the Conversation Module
# This module combines the three components to handle frontend functionalities,
# focusing on the Conversation part of the language learning app.
# It uses Flask to create a simple API server for frontend integration (e.g., via AJAX calls).
# Endpoints:
# - /vocabulary: Generate vocabulary using model.py
# - /grammar: Explain grammar using model.py
# - /conversation: Handle multi-turn conversation using agent.py (which integrates RAG)
# - /rag_query: Direct RAG query for testing
# Run with: python main.py
# Requires: pip install flask
# Access e.g., http://localhost:5000/conversation?user_input=Hola&scenario=restaurant&level=beginner

from flask import Flask, request, jsonify, render_template
import sys
import os
import traceback
import json
from mc_generator import (
    get_Mandarin_MC_question, 
    get_Spanish_MC_question, 
    get_Japanese_MC_question,
    get_German_MC_question, 
    get_French_MC_question
)
from ordering_generator import (
    get_Mandarin_order_question,
    get_Spanish_order_question,
    get_Japanese_order_question,
    get_German_order_question,
    get_French_order_question
)
# Adjust sys.path if files are in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the three modules
from model import generate_vocabulary, explain_grammar, generate_conversation_response, translate_text, call_llm, load_vocabulary_from_file
from rag import RAGSystem, rag_query  # RAGSystem class and query function
from agent import run_agent, AgentState, app_graph  # Agent runner, state, and compiled graph

app = Flask(__name__)

# Initialize RAGSystem globally (load once)
rag_system = RAGSystem()  # Assumes default sources; customize if needed

# Store conversation states (for multi-turn; in production, use session/DB)
conversation_states = {}  # Key: session_id, Value: AgentState

@app.route('/vocabulary', methods=['GET'])
def vocabulary():
    """
    Endpoint for generating vocabulary (from file or model.py)
    Params: language (str), level (str), count (int), category (str optional)
    """
    language = request.args.get('language', 'Spanish')
    level = request.args.get('level', 'beginner')
    count = int(request.args.get('count', 5))  # Changed from 25 to 5
    category = request.args.get('category', None)
    
    # Validate category against allowed categories
    allowed_categories = ["common phrases", "food & dining", "travel", "business", "sports & hobbies"]
    if category and category.lower() not in [c.lower() for c in allowed_categories]:
        category = "common phrases"  # Default to common phrases if invalid
    
    try:
        # First try to load from file
        vocab = load_vocabulary_from_file(language, category)
        
        if vocab:
            # Return vocabulary from file (limited to count)
            return jsonify({"vocabulary": vocab[:count]})
        else:
            # Fallback to model.py generation
            vocab = generate_vocabulary(language, level, count, category)
            return jsonify({"vocabulary": vocab})
            
    except Exception as e:
        print(f"Error in vocabulary: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/vocabulary/generate', methods=['POST'])
def generate_vocabulary_endpoint():
    """
    Generate new vocabulary words using AI and save to file.
    Expects JSON: {language: str, category: str, count: int}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        language = data.get('language', 'Spanish')
        category = data.get('category', 'common phrases')  # Default to common phrases
        count = data.get('count', 5)  # Default to 5 new words
        
        # Validate category
        allowed_categories = ["common phrases", "food & dining", "travel", "business", "sports & hobbies"]
        if category.lower() not in [c.lower() for c in allowed_categories]:
            category = "common phrases"  # Default to common phrases if invalid
        
        # Use model.py to generate new vocabulary
        from model import generate_vocabulary
        
        # Generate new vocabulary
        new_vocab = generate_vocabulary(language, "intermediate", count, category)
        
        # Save to the corresponding language file
        save_vocabulary_to_file(language, new_vocab, category)
        
        return jsonify({
            "vocabulary": new_vocab,
            "message": f"Generated {len(new_vocab)} new words for {category} in {language}"
        })
        
    except Exception as e:
        print(f"Error generating vocabulary: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/vocabulary/categories', methods=['GET'])
def vocabulary_categories():
    """
    Endpoint to get vocabulary categories for a language
    Params: language (str)
    """
    language = request.args.get('language', 'Spanish')
    
    try:
        # Load vocabulary and extract unique categories
        vocab = load_vocabulary_from_file(language)
        categories_count = {}
        
        for item in vocab:
            category = item.get('category', 'general')
            categories_count[category] = categories_count.get(category, 0) + 1
        
        # Format categories for frontend
        categories = [{'name': cat.capitalize(), 'count': count} 
                     for cat, count in categories_count.items()]
        
        return jsonify({"categories": categories})
    except Exception as e:
        print(f"Error getting categories: {str(e)}")
        return jsonify({"categories": []})

def save_vocabulary_to_file(language, vocabulary, category):
    """
    Save generated vocabulary to the language-specific Vocabulary.txt file
    """
    try:
        # Map language names to folder names
        language_folders = {
            'spanish': 'Spanish',
            'french': 'French', 
            'german': 'German',
            'japanese': 'Japanese',
            'mandarin': 'Mandarin',
            'chinese': 'Mandarin'
        }
        
        language_lower = language.lower()
        folder_name = language_folders.get(language_lower, language.capitalize())
        
        # Construct file path
        file_path = os.path.join('language', folder_name, 'Vocabulary.txt')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Format vocabulary items for saving
        vocab_items = []
        for item in vocabulary:
            vocab_item = {
                "word": item.get('word', ''),
                "partOfSpeech": item.get('part_of_speech', ''),
                "definition": item.get('definition', ''),
                "example": item.get('example', ''),
                "category": category
            }
            vocab_items.append(vocab_item)
        
        # Append to file (create if doesn't exist)
        with open(file_path, 'a', encoding='utf-8') as file:
            for item in vocab_items:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(vocab_items)} vocabulary items to {file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving vocabulary to file: {str(e)}")
        return False

@app.route('/multiple_choice', methods=['GET'])
def multiple_choice():
    """
    Endpoint for multiple choice questions
    Params: language (str)
    """
    language = request.args.get('language', 'Mandarin')
    
    try:
        # Map language names to folder names
        language_folders = {
            'spanish': 'Spanish',
            'french': 'French', 
            'german': 'German',
            'japanese': 'Japanese',
            'mandarin': 'Mandarin',
            'chinese': 'Mandarin'
        }
        
        language_lower = language.lower()
        folder_name = language_folders.get(language_lower, language.capitalize())
        
        # Construct file path
        file_path = os.path.join('language', folder_name, 'MC_Question.txt')
        
        questions = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        question_data = json.loads(line)
                        # Convert answer from 1-based to 0-based indexing
                        if 'answer' in question_data:
                            question_data['answer'] = question_data['answer'] - 1
                        questions.append(question_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line}")
                        continue
        
        print(f"Loaded {len(questions)} multiple choice questions from {file_path}")
        return jsonify({"questions": questions})
        
    except FileNotFoundError:
        print(f"Multiple choice file not found: {file_path}")
        return jsonify({"questions": []})
    except Exception as e:
        print(f"Error loading multiple choice questions: {str(e)}")
        return jsonify({"questions": []})
# Add this to main.py after the existing endpoints

@app.route('/multiple_choice/generate', methods=['POST'])
def generate_mc_questions():
    """
    Generate new multiple choice questions using AI.
    Expects JSON: {language: str, count: int}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        language = data.get('language', 'Spanish')
        count = data.get('count', 5)  # Default to 5 new questions
        
        # Map language to the appropriate generation function
        language_functions = {
            'spanish': get_Spanish_MC_question,
            'french': get_French_MC_question,
            'german': get_German_MC_question,
            'japanese': get_Japanese_MC_question,
            'mandarin': get_Mandarin_MC_question,
            'chinese': get_Mandarin_MC_question  # Map Chinese to Mandarin
        }
        
        language_lower = language.lower()
        if language_lower not in language_functions:
            return jsonify({"error": f"Unsupported language: {language}"}), 400
        
        # Generate new questions
        new_questions = language_functions[language_lower]()
        
        return jsonify({
            "questions": new_questions,
            "message": f"Generated {len(new_questions)} new questions for {language}"
        })
        
    except Exception as e:
        print(f"Error generating MC questions: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/ordering_questions', methods=['GET'])
def ordering_questions():
    """
    Endpoint for ordering questions
    Params: language (str)
    """
    language = request.args.get('language', 'Mandarin')
    
    try:
        # Map language names to folder names
        language_folders = {
            'spanish': 'Spanish',
            'french': 'French', 
            'german': 'German',
            'japanese': 'Japanese',
            'mandarin': 'Mandarin',
            'chinese': 'Mandarin'
        }
        
        language_lower = language.lower()
        folder_name = language_folders.get(language_lower, language.capitalize())
        
        # Construct file path
        file_path = os.path.join('language', folder_name, 'Order_Question.txt')
        
        questions = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        question_data = json.loads(line)
                        questions.append(question_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line}")
                        continue
        
        print(f"Loaded {len(questions)} ordering questions from {file_path}")
        return jsonify({"questions": questions})
        
    except FileNotFoundError:
        print(f"Ordering questions file not found: {file_path}")
        return jsonify({"questions": []})
    except Exception as e:
        print(f"Error loading ordering questions: {str(e)}")
        return jsonify({"questions": []})
    
 # Add this to main.py after the existing endpoints

@app.route('/ordering_questions/generate', methods=['POST'])
def generate_ordering_questions():
    """
    Generate new ordering questions using AI.
    Expects JSON: {language: str, count: int}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        language = data.get('language', 'Spanish')
        count = data.get('count', 5)  # Default to 5 new questions
        
        # Map language to the appropriate generation function
        language_functions = {
            'spanish': get_Spanish_order_question,
            'french': get_French_order_question,
            'german': get_German_order_question,
            'japanese': get_Japanese_order_question,
            'mandarin': get_Mandarin_order_question,
            'chinese': get_Mandarin_order_question  # Map Chinese to Mandarin
        }
        
        language_lower = language.lower()
        if language_lower not in language_functions:
            return jsonify({"error": f"Unsupported language: {language}"}), 400
        
        # Generate new questions
        new_questions = language_functions[language_lower]()
        
        return jsonify({
            "questions": new_questions,
            "message": f"Generated {len(new_questions)} new ordering questions for {language}"
        })
        
    except Exception as e:
        print(f"Error generating ordering questions: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500   

@app.route('/grammar', methods=['GET'])
def grammar():
    """
    Endpoint for grammar explanation (from model.py).
    Params: language (str), query (str)
    """
    language = request.args.get('language', 'Spanish')
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        explanation = explain_grammar(language, query)
        return jsonify({"explanation": explanation})
    except Exception as e:
        print(f"Error in grammar: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# In main.py - Add this after the existing endpoints

@app.route('/grammar/progress', methods=['GET'])
def grammar_progress():
    """
    Endpoint to get grammar progress (total questions answered)
    Params: session_id (str)
    """
    session_id = request.args.get('session_id', 'default')
    
    try:
        # Load grammar progress from localStorage or use default
        # In a real app, this would be stored in a database
        grammar_data = {
            "total_answered": 0,
            "ordering_answered": 0,
            "mc_answered": 0
        }
        
        # For prototype, we'll track this client-side and update via this endpoint
        return jsonify(grammar_data)
    except Exception as e:
        print(f"Error getting grammar progress: {str(e)}")
        return jsonify({"total_answered": 0, "ordering_answered": 0, "mc_answered": 0})

@app.route('/translate', methods=['GET'])
def translate():
    """
    Endpoint for translation (from model.py).
    Params: text (str), source_lang (str), target_lang (str)
    """
    text = request.args.get('text', '')
    source_lang = request.args.get('source_lang', 'English')
    target_lang = request.args.get('target_lang', 'Spanish')
    
    if not text:
        return jsonify({"error": "Text parameter is required"}), 400
    
    try:
        translation = translate_text(text, source_lang, target_lang)
        return jsonify({"translation": translation})
    except Exception as e:
        print(f"Error in translate: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/conversation', methods=['GET'])
def conversation():
    """
    Endpoint for conversation response (using agent.py, which integrates RAG).
    Params: user_input (str), scenario (str), level (str), session_id (str optional for multi-turn)
    Returns: response (str), updated_progress (dict), state (for frontend to store session_id)
    """
    user_input = request.args.get('user_input', '')
    scenario = request.args.get('scenario', 'restaurant')
    level = request.args.get('level', 'beginner')
    language = request.args.get('language', 'Spanish')
    session_id = request.args.get('session_id', 'default')  # Use a unique ID from frontend
    
    if not user_input:
        return jsonify({"error": "user_input parameter is required"}), 400
    
    try:
        # Load existing state if multi-turn
        state = conversation_states.get(session_id, None)
        if state is None:
            # Create a new session state when none exists
            state = AgentState(
                messages=[],
                scenario=scenario,
                user_level=level,
                language=language,
                progress={"words_mastered": 0, "accuracy": 0},
                last_correction=""
            )
            print(f"New session created for {session_id}")
        else:
            # If an existing session is reused but the scenario changed,
            # reset the message history to avoid role/context bleed between scenarios.
            prev_scenario = state.get("scenario")
            if prev_scenario != scenario:
                print(f"Session {session_id} scenario changed from {prev_scenario} to {scenario}; resetting history to avoid role bleed.")
                state["messages"] = []

            # Always update scenario, user level, and language to match the current request
            state["scenario"] = scenario
            state["user_level"] = level
            state["language"] = language

        # Run the agent
        updated_state = run_agent(user_input, state)
        
        # Save updated state
        conversation_states[session_id] = updated_state
        
        content = updated_state["messages"][-1]["content"]
        translation = "N/A"
        tips = ""

        # Robust JSON extraction: try multiple strategies to parse assistant output
        parsed = None
        def try_parse(s):
            try:
                return json.loads(s)
            except Exception:
                return None

        # 1) direct parse
        parsed = try_parse(content)

        # 2) sometimes assistant returns a JSON string embedded/escaped; unquote once and parse
        if parsed is None:
            # if content looks like a quoted JSON string, try to unescape it
            if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                try:
                    unquoted = json.loads(content)
                    parsed = try_parse(unquoted)
                except Exception:
                    parsed = None

        # 3) extract the first {...} block and try to parse that
        if parsed is None and '{' in content and '}' in content:
            # Basic first/last brace attempt
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                maybe = content[start:end+1]
                parsed = try_parse(maybe)
            # If that failed, try to find a JSON object containing the key "response" using brace matching
            if parsed is None and '"response"' in content:
                try:
                    idx = content.find('"response"')
                    start = content.rfind('{', 0, idx)
                    if start != -1:
                        # scan forward to find the matching closing brace
                        depth = 0
                        for i in range(start, len(content)):
                            if content[i] == '{':
                                depth += 1
                            elif content[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    maybe = content[start:i+1]
                                    parsed = try_parse(maybe)
                                    break
                except Exception:
                    parsed = None

        # 4) fallback: if parsing failed, leave content as-is and try a heuristic for translation
        if parsed is not None and isinstance(parsed, dict):
            response = parsed.get("response", content)
            translation = parsed.get("translation", "N/A")
            tips = parsed.get("tips", "")
        else:
            # Fallback parsing heuristics: try to split by parentheses for a translation
            response = content
            if '(' in content and ')' in content:
                parts = content.rsplit('(', 1)
                response = parts[0].strip()
                translation = parts[1].rsplit(')', 1)[0].strip() if len(parts) > 1 else "N/A"

        # Post-process fields: unescape unicode escapes if present, normalize quotes
        def unescape_if_needed(s):
            if not isinstance(s, str):
                return s
            # If contains literal backslash-u unicode escapes, attempt to decode
            if '\\u' in s or '\\x' in s:
                try:
                    return s.encode('utf-8').decode('unicode_escape')
                except Exception:
                    pass
            return s

        response = unescape_if_needed(response)
        translation = unescape_if_needed(translation)
        tips = unescape_if_needed(tips)

        # Normalize smart quotes and double double-quotes
        def normalize_quotes(s):
            if not isinstance(s, str):
                return s
            s = s.replace('"', '"').replace('"', '"').replace("''", '"')
            # Fix repeated double quotes
            s = s.replace('""', '"')
            return s.strip()

        response = normalize_quotes(response)
        translation = normalize_quotes(translation)
        tips = normalize_quotes(tips)

        # If language is Mandarin, ensure response contains CJK Unified Ideographs; if not, attempt one retry
        def contains_cjk(s):
            if not isinstance(s, str):
                return False
            for ch in s:
                if '\u4e00' <= ch <= '\u9fff':
                    return True
            return False

        need_retry = False
        req_language = language.lower() if 'language' in locals() and language else 'spanish'
        if parsed is None:
            need_retry = True
        elif req_language in ('mandarin', 'chinese') and not contains_cjk(response):
            need_retry = True

        # Perform a single retry with a clarifying system instruction if needed
        if need_retry:
            try:
                clarifier = {
                    'role': 'system',
                    'content': (
                        "IMPORTANT: You must output strictly valid JSON with keys 'response', 'translation', and 'tips' and nothing else. "
                        "For Mandarin/Chinese, put the reply in Simplified Chinese characters in 'response', the English translation in 'translation', and corrective tips in clear English in 'tips'. "
                        "Do not include any extra text. Return ONLY the JSON object."
                    )
                }
                # Append the clarifier to a copy of the state messages and re-invoke the graph once
                temp_state = updated_state.copy()
                temp_msgs = list(temp_state.get('messages', []))
                temp_msgs.append(clarifier)
                temp_state['messages'] = temp_msgs
                # Invoke the compiled graph directly to get a new assistant response
                retried = app_graph.invoke(temp_state)
                new_content = retried['messages'][-1]['content']
                # Try to parse the new content using the same helper logic
                parsed_retry = None
                try:
                    parsed_retry = json.loads(new_content)
                except Exception:
                    # attempt to extract {...}
                    if '{' in new_content and '}' in new_content:
                        s = new_content[new_content.find('{'): new_content.rfind('}')+1]
                        try:
                            parsed_retry = json.loads(s)
                        except Exception:
                            parsed_retry = None
                if isinstance(parsed_retry, dict):
                    # Handle case where the assistant still returned a JSON string inside one of the fields
                    def extract_maybe_nested(val):
                        if not isinstance(val, str):
                            return val
                        # If looks like JSON object inside string, try unescape and parse
                        candidate = val
                        # common: escaped quotes \" -> replace then try
                        if '\\\"' in candidate or '\\"' in candidate:
                            try_candidate = candidate.replace('\\\"', '\\"').replace('\\"', '\\"')
                            try:
                                # unescape unicode sequences
                                maybe = try_candidate.encode('utf-8').decode('unicode_escape')
                                inner = try_parse(maybe)
                                if isinstance(inner, dict):
                                    return inner
                            except Exception:
                                pass
                        # direct brace content
                        if candidate.strip().startswith('{') and candidate.strip().endswith('}'):
                            p = try_parse(candidate)
                            if isinstance(p, dict):
                                return p
                            try:
                                p2 = try_parse(candidate.encode('utf-8').decode('unicode_escape'))
                                if isinstance(p2, dict):
                                    return p2
                            except Exception:
                                pass
                        return val

                    r_field = extract_maybe_nested(parsed_retry.get('response', response))
                    t_field = extract_maybe_nested(parsed_retry.get('translation', translation))
                    tips_field = extract_maybe_nested(parsed_retry.get('tips', tips))

                    # If extraction returned a dict (nested JSON), map its keys
                    if isinstance(r_field, dict):
                        response = normalize_quotes(unescape_if_needed(r_field.get('response', response)))
                        translation = normalize_quotes(unescape_if_needed(r_field.get('translation', translation)))
                        tips = normalize_quotes(unescape_if_needed(r_field.get('tips', tips)))
                    else:
                        response = normalize_quotes(unescape_if_needed(r_field))
                        translation = normalize_quotes(unescape_if_needed(t_field))
                        tips = normalize_quotes(unescape_if_needed(tips_field))

                    # Replace updated_state with retried so progress etc. reflect retry
                    updated_state = retried
            except Exception as e:
                print('Retry failed:', e)

        # If after retry we still don't have a clean parsed result, fall back to calling the model layer directly
        def looks_valid(resp, trans, tps, lang):
            if not resp:
                return False
            if lang in ('mandarin', 'chinese'):
                return contains_cjk(resp)
            return True

        if not looks_valid(response, translation, tips, req_language):
            try:
                # Build a simple history for the model call from the updated_state messages
                history_msgs = updated_state.get('messages', []) if isinstance(updated_state, dict) else []
                # Call the model layer directly with stricter instructions via model.generate_conversation_response
                fallback_raw = generate_conversation_response(language.title() if language else 'Chinese', scenario, user_input, level, history=history_msgs)
                # Try to parse fallback_raw
                parsed_fb = None
                try:
                    parsed_fb = json.loads(fallback_raw)
                except Exception:
                    # try extracting {...}
                    if '{' in fallback_raw and '}' in fallback_raw:
                        s = fallback_raw[fallback_raw.find('{'): fallback_raw.rfind('}')+1]
                        try:
                            parsed_fb = json.loads(s)
                        except Exception:
                            parsed_fb = None

                if isinstance(parsed_fb, dict):
                    response = normalize_quotes(unescape_if_needed(parsed_fb.get('response', response)))
                    translation = normalize_quotes(unescape_if_needed(parsed_fb.get('translation', translation)))
                    tips = normalize_quotes(unescape_if_needed(parsed_fb.get('tips', tips)))
                else:
                    # Last-resort safe canned Mandarin reply if still invalid
                    if req_language in ('mandarin', 'chinese'):
                        response = '欢迎光临！我们有菜单。您想点什么？'
                        translation = 'Welcome! We have a menu. What would you like to order?'
                        tips = "Use '点菜' to order food or ask '有什么推荐?' for recommendations."
            except Exception as e:
                print('Fallback model call failed:', e)

        # Final attempt: if `response` itself contains a JSON object, extract it
        try:
            if isinstance(response, str) and ('{"response"' in response or "{ \"response\"" in response or '"response"' in response):
                # locate earliest '{' before '"response"'
                idx = None
                if '"response"' in response:
                    pos = response.find('"response"')
                    idx = response.rfind('{', 0, pos)
                if idx is None:
                    idx = response.find('{')
                if idx != -1:
                    depth = 0
                    for i in range(idx, len(response)):
                        if response[i] == '{':
                            depth += 1
                        elif response[i] == '}':
                            depth -= 1
                            if depth == 0:
                                maybe = response[idx:i+1]
                                try:
                                    inner = json.loads(maybe)
                                    if isinstance(inner, dict):
                                        response = normalize_quotes(unescape_if_needed(inner.get('response', response)))
                                        translation = normalize_quotes(unescape_if_needed(inner.get('translation', translation)))
                                        tips = normalize_quotes(unescape_if_needed(inner.get('tips', tips)))
                                except Exception:
                                    # try to decode unicode escapes and retry
                                    try:
                                        cand = maybe.encode('utf-8').decode('unicode_escape')
                                        inner = json.loads(cand)
                                        if isinstance(inner, dict):
                                            response = normalize_quotes(unescape_if_needed(inner.get('response', response)))
                                            translation = normalize_quotes(unescape_if_needed(inner.get('translation', translation)))
                                            tips = normalize_quotes(unescape_if_needed(inner.get('tips', tips)))
                                    except Exception:
                                        pass
                                break
        except Exception:
            pass

        # Additional aggressive extraction: look for literal JSON starting with {"response" or {\"response\"}
        if (not parsed) and '{"response"' in content or '{\\"response\\"' in content:
            try:
                # try to find the substring that starts with either pattern
                idx = content.find('{"response"')
                if idx == -1:
                    idx = content.find('{\\"response\\"')
                if idx != -1:
                    # perform brace matching from idx
                    start = idx
                    depth = 0
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                maybe = content[start:i+1]
                                # attempt to unescape common escapes
                                try_candidates = [maybe, maybe.replace('\\"', '"'), maybe.encode('utf-8').decode('unicode_escape')]
                                for cand in try_candidates:
                                    p = try_parse(cand)
                                    if p:
                                        parsed = p
                                        break
                                break
            except Exception:
                parsed = None

        # If aggressive extraction succeeded, map fields
        if parsed is not None and isinstance(parsed, dict):
            response = parsed.get("response", content)
            translation = parsed.get("translation", "N/A")
            tips = parsed.get("tips", "")

        return jsonify({
            "response": response,
            "translation": translation,
            "tips": tips,
            "progress": updated_state["progress"],
            "session_id": session_id  # Return for frontend to track
        })
    except Exception as e:
        print(f"Error in conversation: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/rag_query', methods=['GET'])
def rag():
    """
    Endpoint for direct RAG query (from rag.py; for testing or standalone use).
    Params: query (str), k (int optional), filter (json str optional, e.g., '{"scenario": "restaurant"}')
    """
    query = request.args.get('query', '')
    k = int(request.args.get('k', 4))
    filter_str = request.args.get('filter', None)
    filter_dict = None
    if filter_str:
        try:
            filter_dict = json.loads(filter_str)
        except:
            return jsonify({"error": "Invalid filter JSON"}), 400
    
    if not query:
        return jsonify({"error": "query parameter is required"}), 400
    
    try:
        response = rag_system.query(query, k=k, filter=filter_dict)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in rag_query: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/conversation/level', methods=['GET'])
def conversation_level():
    """
    Evaluate the user's conversation level based on recent conversation history.
    Params: session_id (str), language (str optional)
    Returns: { conversation_level: 'beginner'|'intermediate'|'advanced', confidence: float (0-1), reason: str }
    """
    session_id = request.args.get('session_id', None)
    language = request.args.get('language', 'Spanish')
    scenario = request.args.get('scenario', None)

    try:
        # If session state exists on server, use its messages; otherwise return unknown
        state = conversation_states.get(session_id)
        messages = []
        if state and isinstance(state, dict):
            messages = state.get('messages', [])
        # Fallback: no server-side history
        if not messages:
            return jsonify({"conversation_level": "unknown", "confidence": 0.0, "reason": "No server-side conversation history available for this session."})

        # Prepare a compact narrative of the last N exchanges (user+ai) to send to the model
        # We'll include up to the last 10 user turns (and surrounding assistant replies) for context.
        # Flatten into alternating speaker lines for easier scoring.
        flattened = []
        count = 0
        # iterate from the end and collect up to 20 messages (approx 10 turns)
        for msg in reversed(messages):
            if count >= 20:
                break
            role = msg.get('role', '')
            content = msg.get('content')
            flattened.append(f"{role.upper()}: {content}")
            count += 1
        flattened.reverse()

        system_prompt = (
            "You are an expert language assessment assistant. Given a short conversation between a learner (USER) and an AI tutor (AI), "
            "assess the learner's overall conversation skill level as one of: beginner, intermediate, or advanced. "
            "Consider vocabulary range, grammatical accuracy, ability to maintain the thread, fluency, and ability to respond appropriately. "
            "Return ONLY a JSON object with keys: 'level' (one of 'beginner'|'intermediate'|'advanced'), 'confidence' (0-1 float), and 'reason' (a 1-2 sentence rationale in English)."
        )

        user_content = "Conversation excerpt:\n" + "\n".join(flattened)
        if scenario:
            user_content = f"Scenario: {scenario}\n" + user_content
        user_content = user_content + f"\nTarget language: {language}." 

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        raw = call_llm(llm_messages, max_tokens=256, temperature=0.0)
        # Try to parse JSON from raw
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # try to extract first {...}
            if '{' in raw and '}' in raw:
                try:
                    s = raw[raw.find('{'): raw.rfind('}')+1]
                    parsed = json.loads(s)
                except Exception:
                    parsed = None

        if not parsed or not isinstance(parsed, dict):
            # best-effort: look for a level word in the raw text
            lvl = 'unknown'
            low = raw.lower()
            if 'beginner' in low:
                lvl = 'beginner'
            elif 'intermediate' in low:
                lvl = 'intermediate'
            elif 'advanced' in low:
                lvl = 'advanced'
            return jsonify({"conversation_level": lvl, "confidence": 0.5, "reason": raw})

        level = parsed.get('level') or parsed.get('conversation_level') or parsed.get('level_label')
        confidence = parsed.get('confidence', 0.0)
        reason = parsed.get('reason', '')
        return jsonify({"conversation_level": level, "confidence": confidence, "reason": reason})
    except Exception as e:
        print(f"Error evaluating conversation level: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/conversation/level_from_history', methods=['POST'])
def conversation_level_from_history():
    """
    Evaluate conversation level from client-sent history.
    Expects JSON body: { history: [{role: 'user'|'ai', content: '...'}], language: 'Spanish', scenario: 'restaurant' }
    Returns same structure as /conversation/level.
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "JSON body required"}), 400

        history = payload.get('history')
        language = payload.get('language', 'Spanish')
        scenario = payload.get('scenario')

        if not history or not isinstance(history, list):
            return jsonify({"error": "history must be a non-empty list of message objects"}), 400

        # Flatten last ~20 messages (approx 10 turns) into compact lines
        flattened = []
        count = 0
        for msg in list(reversed(history)):
            if count >= 20:
                break
            role = (msg.get('role') or '').upper()
            content = msg.get('content') or msg.get('text') or ''
            flattened.append(f"{role}: {content}")
            count += 1
        flattened.reverse()

        system_prompt = (
            "You are an expert language assessment assistant. Given a short conversation between a learner (USER) and an AI tutor (AI), "
            "assess the learner's overall conversation skill level as one of: beginner, intermediate, or advanced. "
            "Consider vocabulary range, grammatical accuracy, ability to maintain the thread, fluency, and ability to respond appropriately. "
            "Return ONLY a JSON object with keys: 'level' (one of 'beginner'|'intermediate'|'advanced'), 'confidence' (0-1 float), and 'reason' (a 1-2 sentence rationale in English)."
        )

        user_content = "Conversation excerpt:\n" + "\n".join(flattened)
        if scenario:
            user_content = f"Scenario: {scenario}\n" + user_content
        user_content = user_content + f"\nTarget language: {language}."

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        raw = call_llm(llm_messages, max_tokens=256, temperature=0.0)

        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            if '{' in raw and '}' in raw:
                try:
                    s = raw[raw.find('{'): raw.rfind('}')+1]
                    parsed = json.loads(s)
                except Exception:
                    parsed = None

        if not parsed or not isinstance(parsed, dict):
            lvl = 'unknown'
            low = raw.lower()
            if 'beginner' in low:
                lvl = 'beginner'
            elif 'intermediate' in low:
                lvl = 'intermediate'
            elif 'advanced' in low:
                lvl = 'advanced'
            return jsonify({"conversation_level": lvl, "confidence": 0.5, "reason": raw})

        level = parsed.get('level') or parsed.get('conversation_level') or parsed.get('level_label')
        confidence = parsed.get('confidence', 0.0)
        reason = parsed.get('reason', '')
        return jsonify({"conversation_level": level, "confidence": confidence, "reason": reason})
    except Exception as e:
        print(f"Error in conversation_level_from_history: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/conversation/reset', methods=['GET'])
def conversation_reset():
    """
    Reset or clear a conversation session on the server side.
    Params: session_id (str), scenario (str optional)
    Returns: { status: 'ok' }
    """
    session_id = request.args.get('session_id', None)
    scenario = request.args.get('scenario', None)

    if not session_id:
        return jsonify({"error": "session_id parameter is required"}), 400

    try:
        if session_id in conversation_states:
            # Remove stored state to fully reset server-side memory for that session
            del conversation_states[session_id]
            print(f"Conversation session {session_id} reset by client request.")
        else:
            print(f"Conversation reset requested for unknown session {session_id}; nothing to delete.")

        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"Error resetting conversation: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    # For demo: Run a simple CLI loop if no Flask, but here we run the server
    app.run(debug=True, port=5000)