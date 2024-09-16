import json
import re
import os

# Get the current script's directory and project root dynamically
project_root = os.path.dirname(os.path.abspath(__file__))

# Define emotion lists
seven_emotions = ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Neutral"]
thirty_four_emotions = ["Afraid", "Angry", "Annoyed", "Anticipating", "Anxious", "Apprehensive", 
                        "Ashamed", "Caring", "Confident", "Content", "Devastated", 
                        "Disappointed", "Disgusted", "Embarrassed", "Excited", "Faithful", 
                        "Furious", "Grateful", "Guilty", "Hopeful", "Impressed", 
                        "Jealous", "Joyful", "Lonely", "Nostalgic", "Prepared", 
                        "Proud", "Sad", "Sentimental", "Surprised", "Terrified", 
                        "Trusting", "정", "한"]

# Create a list of 32 emotions by excluding '정' and '한'
thirty_two_emotions = [emotion for emotion in thirty_four_emotions if emotion not in ["정", "한"]]

# Extract emotions from text (case-insensitive)
def extract_emotions(text, emotion_list, single_emotion=False):
    emotions = []
    for emotion in emotion_list:
        if re.search(r'\b' + re.escape(emotion) + r'\b', text, re.IGNORECASE):
            emotions.append(emotion)
        if single_emotion and emotions:
            return emotions[:1]
    return emotions

# Process listener's response by extracting relevant text
def process_listener_response(response):
    match = re.search(r"'content': (.*?Listener:\s*.*)}", response)
    if match:
        return match.group(1).strip()
    match_content = re.search(r"'content':\s*(.*)}", response)
    if match_content:
        return f"Listener: {match_content.group(1).strip()}"
    return response.strip()

# Clean up response to retain only the 'Listener:' part before '\n\n'
def clean_listener_response(response):
    match = re.search(r'(Listener:.*?)(?:\\n\\n|$)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

# Remove duplicate occurrences of 'Listener:'
def clean_listener_statement(statement):
    return re.sub(r'(Listener:\s*)+', 'Listener: ', statement).strip()

# Process files based on the model and language
def process_file(model_id, lang, lang_key, JeongHan=None):
    model_name = model_id.split("/")[-1]

    # Use relative paths based on project_root
    if JeongHan is None:
        input_file = os.path.join(project_root, 'sample', f'results_{model_name}_{lang}.json')
        output_file = input_file
    else:
        input_file = os.path.join(project_root, 'JeongHan', f'results_{model_name}_{lang}_{JeongHan}.json')
        output_file = input_file

    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"File {input_file} does not exist. Skipping.")
        return

    # Open and process the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for conv_id, conv_data in data.items():
        for scenario in conv_data['scenarios']:
            # Process empathetic response
            if 'empathetic_response' in scenario and scenario['empathetic_response']:
                raw_statement = process_listener_response(str(scenario['empathetic_response']))
                scenario['final_empathetic_statement'] = clean_listener_response(raw_statement)
                del scenario['empathetic_response']

            # Choose appropriate emotion list
            single_emotion = False
            if scenario['scenario'] == "34개의 단일 감정":
                emotions_list = thirty_four_emotions
                single_emotion = True
            elif scenario['scenario'] == "34개의 멀티 감정":
                emotions_list = thirty_four_emotions
            else:
                continue
            
            # Replace 'identified_emotions' with 'emotion inference'
            if 'identified_emotions' in scenario and scenario['identified_emotions']:
                scenario['emotion inference'] = extract_emotions(scenario['identified_emotions']['content'], emotions_list, single_emotion)
                del scenario['identified_emotions']
        
        conv_id_count = len(data)
        print(f"{model_id} - {lang} - {JeongHan}: {conv_id_count} conversation IDs processed.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Processed and saved to {output_file}")

# Count occurrences of 'Listener:' in the text
def count_listeners(text):
    return len(re.findall(r'Listener:', text, re.IGNORECASE))

# Extract text after 'here's' and the first 'Listener:'
def extract_listener_after_heres(text):
    match_heres = re.search(r"here's\s+(.*)", text, re.IGNORECASE | re.DOTALL)
    if match_heres:
        text_after_heres = match_heres.group(1).strip()
        match_listener = re.search(r'\n\nListener:\s*(.*)', text_after_heres, re.DOTALL)
        if match_listener:
            return "Listener: " + match_listener.group(1).strip()
    return text.strip()

# Extract the last occurrence of 'Listener:' in the text
def extract_last_listener(text):
    matches = re.findall(r'Listener:\s*(.*?)(?=(?:Listener:|$))', text, re.DOTALL | re.IGNORECASE)
    if matches:
        return "Listener: " + matches[-1].strip()
    return text.strip()

# Process Claude model output and update empathetic response
def claude_process_json(model_id, lang, lang_key, JeongHan=None):
    model_name = model_id.split("/")[-1]
    if JeongHan is None:
        input_file = os.path.join(project_root, "sample", f"results_{model_name}_{lang}.json")
        output_file = input_file
    else:
        input_file = os.path.join(project_root, "JeongHan", f"results_{model_name}_{lang}_{JeongHan}.json")
        output_file = input_file

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for conv_id, conv_data in data.items():
        for scenario in conv_data['scenarios']:
            single_emotion = False
            if scenario['scenario'] == "34개의 단일 감정":
                emotions_list = thirty_four_emotions
                single_emotion = True
            elif scenario['scenario'] == "34개의 멀티 감정":
                emotions_list = thirty_four_emotions
            else:
                emotions_list = []

            if 'identified_emotions' in scenario and isinstance(scenario['identified_emotions'], dict):
                emotion_text = scenario['identified_emotions'].get('content', '')
                scenario['emotion inference'] = extract_emotions(emotion_text, emotions_list, single_emotion)

            if 'empathetic_response' in scenario and isinstance(scenario['empathetic_response'], dict):
                raw_statement = scenario['empathetic_response'].get('content', '')
                if count_listeners(raw_statement) >= 2:
                    scenario['final_empathetic_statement'] = extract_last_listener(raw_statement)
                else:
                    scenario['final_empathetic_statement'] = raw_statement.strip()

            if 'identified_emotions' in scenario:
                del scenario['identified_emotions']
            if 'empathetic_response' in scenario:
                del scenario['empathetic_response']

        conv_id_count = len(data)
        print(f"{model_id} - {lang} - {JeongHan}: {conv_id_count} conversation IDs processed.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Processed and saved to {output_file}")

# Process all models for the normal configuration
def process_normal():
    model_ids = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]

    for model_id in model_ids:
        for lang, lang_key in [("Korean", "ko_utter"), ("English", "utter")]:
            # process_file(model_id, lang, lang_key, None)
            claude_process_json('claude-3-5-sonnet-20240620', lang, lang_key, None)

# Process all models for JeongHan configuration
def process_jeonghan():
    model_ids = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]

    JeongHan_list = [
        "un_80",
        "kr_80",
        "en_80",
        "simple_80"
    ]

    for model_id in model_ids:
        for lang, lang_key in [("Korean", "ko_utter")]:
            for JeongHan in JeongHan_list:
                process_file(model_id, lang, lang_key, JeongHan)
                claude_process_json('claude-3-5-sonnet-20240620', lang, lang_key, JeongHan)

# Main function to determine processing type and execute the appropriate function
def main(processing_type):
    if processing_type == "normal":
        process_normal()
    elif processing_type == "jeonghan":
        process_jeonghan()
    else:
        print("Invalid processing type. Choose 'normal' or 'jeonghan'.")

if __name__ == "__main__":
    main('normal')
