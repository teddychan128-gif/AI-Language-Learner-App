import os
import re
import json
from openai import OpenAI

# Load the OpenAI client with the specified base URL and API key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-api-key",
)

# Default model to use
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
# Specific model for vocabulary generation
VOCABULARY_MODEL = "openai/gpt-4o-mini"

# Enhanced system prompt templates for different functionalities with better language support
SYSTEM_PROMPTS = {
    "vocabulary_generation": {
        "mandarin": """
Please generate 5 vocabulary words for beginners to learn Mandarin.
The category should be {category} (one of: common phrases, food & dining, travel, business, sports & hobbies).
Use the following format for each entry:
{{ 
    "word": "喜欢", 
    "partOfSpeech": "verb", 
    "category": "{category}", 
    "definition": "to like", 
    "example": "这个房子很大。 (Zhège fángzi hěn dà.) - This house is very big." 
}}
Ensure that the words are provided in Simplified Chinese.
The example sentence should include a Mandarin sentence and an English translation that conveys the same meaning.
Only provide the JSON array format, no additional text or explanations.
Do not give me words that already appear in the list: {previous_words}.
Return ONLY a valid JSON array with 5 items.
""",
        "spanish": """
Please generate 5 vocabulary words for beginners to learn Spanish.
The category should be {category} (one of: common phrases, food & dining, travel, business, sports & hobbies).
Use the following format for each entry:
{{ 
    "word": "example", 
    "partOfSpeech": "verb", 
    "category": "{category}", 
    "definition": "english definition", 
    "example": "Spanish sentence - English translation"
}}
Ensure that the words are provided in Spanish.
The example sentence should include a Spanish sentence and an English translation that conveys the same meaning.
Only provide the JSON array format, no additional text or explanations.
Do not give me words that already appear in the list: {previous_words}.
Return ONLY a valid JSON array with 5 items.
""",
        "japanese": """
Please generate 5 vocabulary words for beginners to learn Japanese.
The category should be {category}.
Use the following format for each entry:
{{ 
    "word": "example", 
    "partOfSpeech": "verb", 
    "category": "{category}", 
    "definition": "english definition", 
    "example": "japanese sentence - english translation"
}}
Ensure that the words are provided in Japanese.
The example sentence should include a Japanese sentence and an English translation that conveys the same meaning.
Only provide the JSON array format, no additional text or explanations.
Do not give me words that already appear in the list: {previous_words}.
Return ONLY a valid JSON array with 5 items.
""",
        "german": """
Please generate 5 vocabulary words for beginners to learn German.
The category should be {category}.
Use the following format for each entry:
{{ 
    "word": "example", 
    "partOfSpeech": "verb", 
    "category": "{category}", 
    "definition": "english definition", 
    "example": "German sentence - english translation"
}}
Ensure that the words are provided in German.
The example sentence should include a German sentence and an English translation that conveys the same meaning.
Only provide the JSON array format, no additional text or explanations.
Do not give me words that already appear in the list: {previous_words}.
Return ONLY a valid JSON array with 5 items.
""",
        "french": """
Please generate 5 vocabulary words for beginners to learn French.
The category should be {category}.
Use the following format for each entry:
{{ 
    "word": "example", 
    "partOfSpeech": "verb", 
    "category": "{category}", 
    "definition": "english definition", 
    "example": "French sentence - english translation"
}}
Ensure that the words are provided in French.
The example sentence should include a French sentence and an English translation that conveys the same meaning.
Only provide the JSON array format, no additional text or explanations.
Do not give me words that already appear in the list: {previous_words}.
Return ONLY a valid JSON array with 5 items.
"""
    },
    "grammar_explanation": """
You are a grammar expert. Explain grammar rules clearly for language learners.
Provide explanations, examples, and corrections based on the query and target language.
For each language, focus on the most relevant grammar concepts:
- Spanish: verb conjugation, gender agreement, subjunctive mood
- French: verb conjugation, gender agreement, subjunctive mood
- German: cases (nominative, accusative, dative, genitive), word order, verb placement
- Japanese: particles, verb conjugation, sentence structure, honorifics
- Mandarin: word order, measure words, tones, question formation

Provide clear explanations with examples in the target language and English translations.
""",
    "conversation_response": """
You are an AI conversational partner for language practice.
You are role-playing in the specified scenario.
Respond in the target language, keep responses natural and at the user's level.

After your response, provide an English translation in parentheses or a separate `translation` field.
If the user's input has errors, provide a short corrective feedback message.

Important: Always write any corrective feedback or "tips" in clear English, regardless of the target language. Keep feedback concise (1-2 sentences) and focused on the current user's level (vocabulary for beginner, grammar for intermediate, fluency for advanced).

Additional strict formatting rules:
- Return strictly valid JSON with keys `response`, `translation`, and `tips` and nothing else. Do NOT include extra prose outside the JSON.
- For Mandarin/Chinese: `response` must be in Simplified Chinese characters; `translation` in English; include pinyin in parentheses when showing pronunciation examples (but do not put pinyin in the `response` field itself).
- For Japanese: `response` should be in natural Japanese (kanji/kana mix appropriate for the level); `translation` in English.
- For other languages: `response` should be in the correct target language script, `translation` in English, `tips` in English.
""",
    "translation": """
You are a professional translator. Translate the given text accurately from source to target language.
Consider cultural context and natural phrasing in the target language.
For language learning contexts, provide clear, educational translations that help learners understand structure and usage.
"""
}

def call_llm(messages, model=DEFAULT_MODEL, max_tokens=1024, temperature=0.1):
    """
    General function to call the LLM with a list of messages.
    
    Args:
        messages (list): List of message dictionaries [{'role': 'system/user/assistant', 'content': 'text'}]
        model (str): Model name to use
        max_tokens (int): Maximum tokens in response
        temperature (float): Sampling temperature
    
    Returns:
        str: The content of the assistant's response
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = completion.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty response from LLM")
        return content
    except Exception as e:
        raise ValueError(f"Error calling LLM: {str(e)}")

def clean_json_response(response):
    """
    Clean LLM response to extract valid JSON array.
    Handles common issues like extra text, markdown, missing brackets, or comma errors.
    
    Args:
        response (str): Raw LLM response
    
    Returns:
        str: Cleaned JSON string
    """
    response = re.sub(r'^.*?(?=\[)', '', response, flags=re.DOTALL).strip()
    if ']' not in response:
        response += ']'
    else:
        response = re.sub(r'\].*?$', ']', response, flags=re.DOTALL)
    response = re.sub(r'```json\s*|\s*```', '', response)
    response = re.sub(r'\n\s*', ' ', response).strip()
    if response.endswith(',]'):
        response = response[:-1] + ']'
    response = re.sub(r'\.\s*\}$', '}', response)
    return response

def load_vocabulary_from_file(language, category=None):
    """
    Load vocabulary from language-specific Vocabulary.txt file
    """
    # Map language names to folder names - fix Chinese to Mandarin mapping
    language_folders = {
        'spanish': 'Spanish',
        'french': 'French', 
        'german': 'German',
        'japanese': 'Japanese',
        'mandarin': 'Mandarin',
        'chinese': 'Mandarin'  # Map Chinese to Mandarin folder
    }
    
    language_lower = language.lower()
    folder_name = language_folders.get(language_lower, language.capitalize())
    
    # Construct file path
    file_path = os.path.join('language', folder_name, 'Vocabulary.txt')
    
    try:
        vocabulary_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        vocab_item = json.loads(line)
                        # Apply category filter if specified (case-insensitive)
                        if category:
                            item_category = vocab_item.get('category', '').lower().strip()
                            filter_category = category.lower().strip()
                            if item_category != filter_category:
                                continue
                                
                        formatted_item = {
                            'word': vocab_item.get('word', ''),
                            'part_of_speech': vocab_item.get('partOfSpeech', ''),
                            'definition': vocab_item.get('definition', ''),
                            'example': vocab_item.get('example', ''),
                            'category': vocab_item.get('category', 'common phrases')
                        }
                        vocabulary_list.append(formatted_item)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line}")
                        continue
        
        print(f"Loaded {len(vocabulary_list)} vocabulary items from {file_path}" + 
              (f" (category: {category})" if category else ""))
        return vocabulary_list
        
    except FileNotFoundError:
        print(f"Vocabulary file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading vocabulary from {file_path}: {str(e)}")
        return []

# In model.py - Update the generate_vocabulary function and system prompts

# Add the specific categories at the top
VOCABULARY_CATEGORIES = ["common phrases", "food & dining", "travel", "business", "sports & hobbies"]

def generate_vocabulary(language, level="beginner", count=5, category=None):
    """
    Generate vocabulary words using AI with language-specific prompts
    Uses openai/gpt-4o-mini model specifically for vocabulary generation
    """
    language_lower = language.lower()
    
    # Validate and set default category
    if not category or category.lower() not in [c.lower() for c in VOCABULARY_CATEGORIES]:
        category = "common phrases"  # Default category
    
    # Get the appropriate prompt for the language
    if language_lower in SYSTEM_PROMPTS["vocabulary_generation"]:
        system_prompt = SYSTEM_PROMPTS["vocabulary_generation"][language_lower]
    else:
        # Fallback to Spanish prompt for unknown languages
        system_prompt = SYSTEM_PROMPTS["vocabulary_generation"]["spanish"]
    
    # Load existing vocabulary to avoid duplicates
    existing_vocab = load_vocabulary_from_file(language, category)
    previous_words = [item['word'] for item in existing_vocab]
    
    # Format the prompt with actual values
    formatted_prompt = system_prompt.format(
        category=category,
        previous_words=previous_words
    )
    
    messages = [
        {"role": "system", "content": formatted_prompt}
    ]
    
    # Use openai/gpt-4o-mini specifically for vocabulary generation
    response = call_llm(messages, model=VOCABULARY_MODEL, temperature=0.1)
    print(f"Raw LLM Response for {language} vocabulary: {response}")
    
    cleaned = clean_json_response(response)
    print(f"Cleaned JSON: {cleaned}")
    
    try:
        vocab_list = json.loads(cleaned)
        if not isinstance(vocab_list, list):
            raise ValueError("Response is not a JSON array")
        
        # Validate and clean each vocabulary item
        validated_vocab = []
        for item in vocab_list:
            if isinstance(item, dict) and all(key in item for key in ['word', 'partOfSpeech', 'definition', 'example']):
                validated_vocab.append({
                    'word': item['word'],
                    'part_of_speech': item['partOfSpeech'],
                    'definition': item['definition'],
                    'example': item['example'],
                    'category': category  # Use the validated category
                })
        
        return validated_vocab
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse vocabulary JSON from LLM response. Raw: {response}. Cleaned: {cleaned}. Error: {str(e)}")

# Update the save_vocabulary_to_file function to handle Chinese->Mandarin mapping
def save_vocabulary_to_file(language, vocabulary, category="common phrases"):
    """
    Save generated vocabulary to the language-specific Vocabulary.txt file
    """
    try:
        # Map language names to folder names - fix Chinese to Mandarin mapping
        language_folders = {
            'spanish': 'Spanish',
            'french': 'French', 
            'german': 'German',
            'japanese': 'Japanese',
            'mandarin': 'Mandarin',
            'chinese': 'Mandarin'  # Map Chinese to Mandarin folder
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

    
def explain_grammar(language, query):
    """
    Explain grammar based on user query.
    
    Args:
        language (str): Target language
        query (str): Grammar question or sentence to correct
    
    Returns:
        str: Explanation text
    """
    # Enhanced prompt with language-specific context
    user_content = f"Language: {language}\nGrammar Query: {query}\n\nPlease provide a clear explanation with examples in {language} and English translations."
    
    # Add language-specific guidance
    if language.lower() in ['japanese', 'mandarin', 'chinese']:
        user_content += " For East Asian languages, focus on writing systems, particles/measure words, and sentence structure."
    elif language.lower() in ['spanish', 'french', 'italian']:
        user_content += " For Romance languages, focus on verb conjugation, gender agreement, and common grammatical structures."
    elif language.lower() in ['german']:
        user_content += " For German, focus on cases, word order, and compound nouns."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["grammar_explanation"]},
        {"role": "user", "content": user_content}
    ]
    
    return call_llm(messages, temperature=0.3)  # Slightly higher temperature for creative explanations

def generate_conversation_response(language, scenario, user_message, level="beginner", history=[]):
    """
    Generate AI response in a conversation simulation.
    
    Args:
        language (str): Target language
        scenario (str): Conversation scenario (e.g., "restaurant")
        user_message (str): User's latest message
        level (str): User's level
        history (list): Previous messages for context
    
    Returns:
        str: AI response with translation and optional feedback
    """
    system_content = SYSTEM_PROMPTS["conversation_response"]
    
    # Enhanced scenario and language context
    scenario_contexts = {
        "restaurant": "You are a waiter/waitress. Handle ordering, menu questions, and customer service.",
        "hotel": "You are a hotel receptionist. Handle check-in, room requests, and hotel services.",
        "shopping": "You are a shop assistant. Help with products, sizes, prices, and recommendations.",
        "doctor": "You are a doctor. Discuss symptoms, give advice, and explain medical terms simply.",
        "interview": "You are a job interviewer. Ask about experience, skills, and career goals."
    }
    
    scenario_context = scenario_contexts.get(scenario, "You are having a conversation in the given scenario.")
    system_content += f"\nScenario: {scenario}. {scenario_context} User's level: {level}. Target language: {language}."
    
    # Adjust complexity based on level
    if level == "beginner":
        system_content += " Use simple vocabulary and short sentences. Focus on basic communication."
    elif level == "intermediate":
        system_content += " Use more varied vocabulary and complex sentence structures. Include some idiomatic expressions."
    else:  # advanced
        system_content += " Use sophisticated vocabulary, complex grammar, and natural conversational flow."
    
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    return call_llm(messages, temperature=0.7)  # Higher temperature for more varied conversation

def translate_text(text, source_lang="English", target_lang="Spanish"):
    """
    Translate text between languages.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language
        target_lang (str): Target language
    
    Returns:
        str: Translated text
    """
    user_content = f"Translate the following text from {source_lang} to {target_lang}:\n\n\"{text}\"\n\n"
    
    # Add translation guidance for language learning context
    if target_lang in ["Japanese", "Mandarin", "Chinese"]:
        user_content += "For language learning purposes, provide a natural translation that maintains the original meaning while being appropriate for the target language's cultural context."
    else:
        user_content += "Provide a clear, accurate translation suitable for language learners."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["translation"]},
        {"role": "user", "content": user_content}
    ]
    
    return call_llm(messages, temperature=0.1)  # Low temperature for accurate translation
 
# Enhanced example usage with multiple languages
if __name__ == "__main__":
    # Test vocabulary generation for different languages
    languages = ["Spanish", "French", "German", "Japanese", "Mandarin"]
    
    for lang in languages:
        print(f"\n=== Testing {lang} Vocabulary ===")
        try:
            vocab = generate_vocabulary(lang, count=3)
            print(f"Generated {lang} Vocabulary:")
            print(json.dumps(vocab, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error generating {lang} vocabulary: {str(e)}")
    
    # Test conversation response
    print("\n=== Testing Conversation Response ===")
    response = generate_conversation_response("Spanish", "restaurant", "Sí, tengo una reserva a nombre de Smith.")
    print("Conversation Response:")
    print(response)
    
    # Test grammar explanation
    print("\n=== Testing Grammar Explanation ===")
    explanation = explain_grammar("Japanese", "When should I use は vs が particles?")
    print("Grammar Explanation:")
    print(explanation)
