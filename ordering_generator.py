# ordering_generator.py
import os
import json
import re
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def clean_json_response(response):
    """
    Clean LLM response to extract valid JSON array.
    More careful approach to preserve the actual JSON structure.
    """
    print(f"Original response: {response}")
    
    # First, try to find a complete JSON array
    json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
        print(f"Found JSON array: {cleaned}")
        return cleaned
    
    # If no complete array found, try to extract individual objects and build array
    json_objects = re.findall(r'\{\s*"[^"]*"[^}]*\}', response)
    if json_objects:
        print(f"Found {len(json_objects)} JSON objects")
        # Build a proper JSON array
        cleaned = '[' + ','.join(json_objects) + ']'
        print(f"Built JSON array: {cleaned}")
        return cleaned
    
    # If still no luck, try the original cleaning method but less aggressively
    cleaned = response.strip()
    
    # Remove markdown code blocks if present
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'\s*```', '', cleaned)
    
    # Remove any text before the first [ and after the last ]
    if '[' in cleaned and ']' in cleaned:
        start = cleaned.find('[')
        end = cleaned.rfind(']') + 1
        cleaned = cleaned[start:end]
    
    print(f"Final cleaned response: {cleaned}")
    return cleaned

def get_ordering_questions(language, output_dir, prompt_template, previous_questions):
    """
    Generic function to generate ordering questions for any language.
    """
    output_file = os.path.join(output_dir, 'Order_Question.txt')
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.format(previous_questions=previous_questions)
                }
            ],
            temperature=0.1
        )
        
        response_content = completion.choices[0].message.content
        print(f"Raw response for {language}:")
        print(response_content)
        print("=" * 50)
        
        cleaned_response = clean_json_response(response_content)
        
        try:
            result_array = json.loads(cleaned_response)
            if not isinstance(result_array, list):
                raise ValueError("Response is not a JSON array")
            
            # Validate each question object
            validated_questions = []
            for item in result_array:
                if isinstance(item, dict):
                    # Ensure all required fields are present and properly formatted
                    validated_item = {
                        "question": item.get("question", []),
                        "answer": str(item.get("answer", "")).strip(),
                        "explain": str(item.get("explain", "")).strip()
                    }
                    
                    # Only add if all required fields are present and valid
                    if (validated_item["question"] and 
                        len(validated_item["question"]) >= 3 and  # At least 3 words
                        validated_item["answer"] and 
                        validated_item["explain"]):
                        validated_questions.append(validated_item)
                    else:
                        print(f"Invalid question skipped: {validated_item}")
            
            if not validated_questions:
                raise ValueError("No valid questions found in response")
                
            result_array = validated_questions
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            print(f"Attempting to fix malformed JSON...")
            
            # More robust JSON extraction
            json_objects = []
            
            # Try to find complete JSON objects with all required fields
            pattern = r'\{\s*"question"\s*:\s*\[[^]]*\]\s*,\s*"answer"\s*:\s*"[^"]*"\s*,\s*"explain"\s*:\s*"[^"]*"\s*\}'
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            
            for match in matches:
                try:
                    item = json.loads(match)
                    json_objects.append(item)
                except:
                    continue
            
            if json_objects:
                result_array = []
                for item in json_objects:
                    validated_item = {
                        "question": item.get("question", []),
                        "answer": str(item.get("answer", "")).strip(),
                        "explain": str(item.get("explain", "")).strip()
                    }
                    result_array.append(validated_item)
                print(f"Recovered {len(result_array)} questions from malformed JSON")
            else:
                raise ValueError(f"Failed to parse {language} ordering questions JSON. Error: {str(e)}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Append new items to existing file
        with open(output_file, 'a', encoding='utf-8') as f:
            for item in result_array:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Successfully appended {len(result_array)} new {language} ordering questions")
        return result_array
        
    except Exception as e:
        print(f"Error generating {language} ordering questions: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def get_previous_ordering_questions(output_file):
    """Helper function to load previous questions from file"""
    previous_questions = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    try:
                        item_dict = json.loads(line.strip())
                        previous_questions.append(item_dict['question'])
                    except:
                        print(f"Could not parse line: {line}")
    return previous_questions

def get_Mandarin_order_question():
    """
    Generate Mandarin word order questions for beginners learning Chinese grammar.
    """
    output_dir = 'language/Mandarin'
    output_file = os.path.join(output_dir, 'Order_Question.txt')
    previous_questions = get_previous_ordering_questions(output_file)
    
    prompt_template = """請給我五題中文詞語排序題，還有答案是完整句子，format is as below:
{{
"question": ["词语1", "词语2", "词语3", "词语4","词语5"],
"answer": "一句通顺的句子, 不要句號",
"explain": "语法结构说明 (English) "
}}，词语1-5是打亂的，把句子裏面的所有字词分成五個词语，句子必須通顺,词语和句子用簡體，請不要重覆以下題目:{previous_questions}，输出最终的、有效的JSON数组，不要有任何其他额外的文本、解释或Markdown代码块标记。"""
    
    return get_ordering_questions("Mandarin", output_dir, prompt_template, previous_questions)

def get_Spanish_order_question():
    """
    Generate Spanish word order questions.
    """
    output_dir = 'language/Spanish'
    output_file = os.path.join(output_dir, 'Order_Question.txt')
    previous_questions = get_previous_ordering_questions(output_file)
    
    prompt_template = """Please provide five Spanish word ordering questions, along with answers that form complete sentences. The format should be as follows:

{{
"question": ["word1", "word2", "word3", "word4", "word5"],
"answer": "a correctly ordered sentence without a period, using the word inside the question list, no need captial letter for the first word",
"explain": "Grammar structure explanation (English)"
}}

All words inside the sentence should be include into the question word 1-5. The words and sentences should be in Spanish, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""
    
    return get_ordering_questions("Spanish", output_dir, prompt_template, previous_questions)

def get_Japanese_order_question():
    """Generate Japanese word order questions."""
    output_dir = 'language/Japanese'
    output_file = os.path.join(output_dir, 'Order_Question.txt')
    previous_questions = get_previous_ordering_questions(output_file)
    
    prompt_template = f""""以下の形式で、日本語のフレーズ順序に関する問題を5題提供してください。各問題には完全な文が含まれ、文内のすべてのフレーズは質問のフレーズリストに使用されなければなりません。フォーマットは次のとおりです：

{{
"question": ["フレーズ1", "フレーズ2", "フレーズ3", "フレーズ4", "フレーズ5"],
"answer": "正しく順序付けられた文（句点なし）で、質問リスト内のフレーズを使用",
"explain": "文法構造の説明（英語）"
}}

文の中に含まれるすべてのフレーズは、質問のフレーズ1-5に含める必要があります。フレーズと文は日本語で書かれ、{previous_questions} に提供された質問を繰り返さないでください。最終的な有効なJSON配列を出力し、追加のテキストや説明、マークダウンコードブロックのマークを含めないでください。

"""
    
    return get_ordering_questions("Japanese", output_dir, prompt_template, previous_questions)

def get_German_order_question():
    """Generate German word order questions."""
    output_dir = 'language/German'
    output_file = os.path.join(output_dir, 'Order_Question.txt')
    previous_questions = get_previous_ordering_questions(output_file)
    
    prompt_template = """Please provide five German word ordering questions, along with answers that form complete sentences. The format should be as follows:

{{
"question": ["word1", "word2", "word3", "word4", "word5"],
"answer": "a correctly ordered sentence without a period, using the word inside the question list, no need captial letter for the first word",
"explain": "Grammar structure explanation (English)"
}}

All words inside the sentence should be include into the question word 1-5. The words and sentences should be in German, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""
    
    return get_ordering_questions("German", output_dir, prompt_template, previous_questions)

def get_French_order_question():
    """Generate French word order questions."""
    output_dir = 'language/French'
    output_file = os.path.join(output_dir, 'Order_Question.txt')
    previous_questions = get_previous_ordering_questions(output_file)
    
    prompt_template = """Please provide five French word ordering questions, along with answers that form complete sentences. The format should be as follows:

{{
"question": ["word1", "word2", "word3", "word4", "word5"],
"answer": "a correctly ordered sentence without a period, using the word inside the question list, no need captial letter for the first word",
"explain": "Grammar structure explanation (English)"
}}

All words inside the sentence should be include into the question word 1-5. The words and sentences should be in French, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""
    
    return get_ordering_questions("French", output_dir, prompt_template, previous_questions)

# Test function
if __name__ == "__main__":
    # Test one language
    questions = get_Spanish_order_question()
    print(f"Generated {len(questions)} Spanish ordering questions")
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"Answer: {q['answer']}")
        print(f"Explanation: {q['explain']}")
        print("---")