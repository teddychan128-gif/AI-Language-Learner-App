# mc_generator.py
import os
import json
import re
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-api-key",
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

def get_mc_questions(language, output_dir, prompt_template, previous_questions):
    """
    Generic function to generate multiple choice questions for any language.
    """
    output_file = os.path.join(output_dir, 'MC_Question.txt')
    
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
                        "question": str(item.get("question", "")).strip(),
                        "options": [str(opt).strip() for opt in item.get("options", []) if str(opt).strip()],
                        "answer": int(item.get("answer", 1)) - 1,  # Convert to 0-based index (subtract 1)
                        "explain": str(item.get("explain", "")).strip()
                    }
                    
                    # Only add if all required fields are present and valid
                    if (validated_item["question"] and 
                        len(validated_item["options"]) == 4 and 
                        0 <= validated_item["answer"] <= 3):
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
            pattern = r'\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"options"\s*:\s*\[[^]]*\]\s*,\s*"answer"\s*:\s*\d+\s*,\s*"explain"\s*:\s*"[^"]*"\s*\}'
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
                        "question": str(item.get("question", "")).strip(),
                        "options": [str(opt).strip() for opt in item.get("options", [])],
                        "answer": int(item.get("answer", 1)) - 1,  # Convert to 0-based index
                        "explain": str(item.get("explain", "")).strip()
                    }
                    result_array.append(validated_item)
                print(f"Recovered {len(result_array)} questions from malformed JSON")
            else:
                raise ValueError(f"Failed to parse {language} MC questions JSON. Error: {str(e)}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Append new items to existing file
        with open(output_file, 'a', encoding='utf-8') as f:
            for item in result_array:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Successfully appended {len(result_array)} new {language} questions")
        return result_array
        
    except Exception as e:
        print(f"Error generating {language} questions: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def get_previous_questions(output_file):
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

def get_Mandarin_MC_question():
    """
    Generate Mandarin multiple choice questions for beginners learning Chinese grammar.
    """
    output_dir = 'language/Mandarin'
    output_file = os.path.join(output_dir, 'MC_Question.txt')
    previous_questions = get_previous_questions(output_file)
    
    prompt_template = """請給我五題多選項題，關於中文文法，給我選項以及正確答案，一共要四個選項，題目是填充題，選項是四個詞語，只有一個正確答案，不要固定答案是同一個選項數目，答案選項有1/2/3/4，題目文法必須正確，format is as below:
{{
"question": "題目",
"options":["選項1","選項2","選項3","選項4"],
"answer": [1/2/3/4，代表選項1/2/3/4],
"explain": "English explain"
}}，題目和選項用簡體，請不要重覆以下題目:{previous_questions}，输出最终的、有效的JSON数组，不要有任何其他额外的文本、解释或Markdown代码块标记。"""
    
    return get_mc_questions("Mandarin", output_dir, prompt_template, previous_questions)

def get_Spanish_MC_question():
    """
    Generate Spanish multiple choice questions.
    """
    output_dir = 'language/Spanish'
    output_file = os.path.join(output_dir, 'MC_Question.txt')
    previous_questions = get_previous_questions(output_file)
    
    prompt_template = """Please provide five multiple-choice questions regarding Spanish grammar. Each question should be a fill-in-the-blank type with four options, where only one option is correct. The options should be words that can fit into the blanks of the questions, and the correct answer option should be randomized e.g question 1:2 question2:4. The answer options should be labeled as 1, 2, 3, or 4. The sentences in the questions must be grammatically correct in Spanish. The format should be as follows:

{{
"question": "Question",
"options": [Option1, Option2, Option3, Option4],
"answer": [1/2/3/4, representing the selected option],
"explain": "English explanation"
}}

The questions and options should be in Spanish, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""

    return get_mc_questions("Spanish", output_dir, prompt_template, previous_questions)

def get_Japanese_MC_question():
    """Generate Japanese multiple choice questions."""
    output_dir = 'language/Japanese'
    output_file = os.path.join(output_dir, 'MC_Question.txt')
    previous_questions = get_previous_questions(output_file)
    
    prompt_template = """Please provide five multiple-choice questions regarding Japanese grammar. Each question should be a fill-in-the-blank type with four options, where only one option is correct. The options should be words that can fit into the blanks of the questions, and the correct answer option should be randomized e.g question 1:3 question2:1. The answer options should be labeled as 1, 2, 3, or 4. The sentences in the questions must be grammatically correct in Japanese. The format should be as follows:

{{
"question": "Question",
"options": [Option1, Option2, Option3, Option4],
"answer": [1/2/3/4, representing the selected option],
"explain": "English explanation"
}}

The questions and options should be in Japanese, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""
    
    return get_mc_questions("Japanese", output_dir, prompt_template, previous_questions)

def get_German_MC_question():
    """Generate German multiple choice questions."""
    output_dir = 'language/German'
    output_file = os.path.join(output_dir, 'MC_Question.txt')
    previous_questions = get_previous_questions(output_file)
    
    prompt_template = """Please provide five multiple-choice questions regarding German grammar. Each question should be a fill-in-the-blank type with four options, where only one option is correct. The options should be words that can fit into the blanks of the questions, and the correct answer option should be randomized e.g question 1:3 question2:1. The answer options should be labeled as 1, 2, 3, or 4. The sentences in the questions must be grammatically correct in German. The format should be as follows:

{{
"question": "Question",
"options": [Option1, Option2, Option3, Option4],
"answer": [1/2/3/4, representing the selected option],
"explain": "English explanation"
}}

The questions and options should be in German, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""
    
    return get_mc_questions("German", output_dir, prompt_template, previous_questions)

def get_French_MC_question():
    """Generate French multiple choice questions."""
    output_dir = 'language/French'
    output_file = os.path.join(output_dir, 'MC_Question.txt')
    previous_questions = get_previous_questions(output_file)
    
    prompt_template = """Please provide five multiple-choice questions regarding French grammar. Each question should be a fill-in-the-blank type with four options, where only one option is correct. The options should be words that can fit into the blanks of the questions, and the correct answer option should be randomized e.g question 1:3 question2:1. The answer options should be labeled as 1, 2, 3, or 4. The sentences in the questions must be grammatically correct in French. The format should be as follows:

{{
"question": "Question",
"options": [Option1, Option2, Option3, Option4],
"answer": [1/2/3/4, representing the selected option],
"explain": "English explanation"
}}

The questions and options should be in French, and please do not repeat the questions provided in: {previous_questions}. Output a final, valid JSON array without any additional text, explanations, or Markdown code block markings."""
    
    return get_mc_questions("French", output_dir, prompt_template, previous_questions)

# Test function
if __name__ == "__main__":
    # Test one language
    questions = get_German_MC_question()
    print(f"Generated {len(questions)} German questions")
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"Options: {q['options']}")
        print(f"Answer: {q['answer']}")
        print(f"Explanation: {q['explain']}")
        print("---")
