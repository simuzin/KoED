import os
import json
import re

### Post-processing to ensure scores are stored as integers ###
def sanitize_filename(filename):
    # Replace characters not allowed in filenames with '_'
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    modified = False
    for scenario in data.values():
        if isinstance(scenario, dict):
            for emotion_data in scenario.values():
                if isinstance(emotion_data, dict) and "scores" in emotion_data and "evaluations" in emotion_data:
                    scores = emotion_data["scores"]
                    evaluations = emotion_data["evaluations"]
                    
                    for criterion, score in scores.items():
                        if score == "Error":
                            evaluation = evaluations.get(criterion, "")
                            # Regular expression to capture both '** number' and 'number **' formats
                            match = re.search(r'\*\*? (\d+)|(\d+)\*\*?', evaluation)
                            if match:
                                # Handle different match groups for both formats
                                new_score = int(match.group(1) or match.group(2))
                                scores[criterion] = new_score
                                modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Updated file: {os.path.basename(file_path)}")

def process_all_files(output_directory, models, languages):
    # Iterate over each model and language combination
    for model in models:
        for language in languages:
            # Create file path for each model and language
            model_dir = os.path.join(output_directory, sanitize_filename(model), sanitize_filename(language))
            file_name = f"{sanitize_filename(model)}_{sanitize_filename(language)}_evaluation.json"
            file_path = os.path.join(model_dir, file_name)
            
            # Process file if it exists
            if os.path.exists(file_path):
                process_json_file(file_path)
            else:
                print(f"File not found: {file_path}")

# Main execution
if __name__ == "__main__":
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the output directory relative to the script's location
    output_directory = os.path.join(current_dir, '..', '..', 'output', 'eval_results', 'sample')

    models = [
        "claude-3-5-sonnet-20240620",
        "Meta-Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Qwen2-7B-Instruct",
        "EXAONE-3.0-7.8B-Instruct"
    ]
    languages = ["Korean", "English"]

    process_all_files(output_directory, models, languages)
