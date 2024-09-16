import os
import openai
import json
from tqdm import tqdm
import time
import re

# Set OpenAI API keys (researcher-specific)
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
openai.api_key = OPENAI_API_KEY

# Load JSON data from the specified file path
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Sanitize the filename to make it safe for saving
def sanitize_filename(name):
    # Replace invalid characters in the filename
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# Save evaluation results to a file
def save_evaluation_to_file(results, output_directory, model_name, language):
    model_dir = os.path.join(output_directory, sanitize_filename(model_name), sanitize_filename(language))
    os.makedirs(model_dir, exist_ok=True)

    # Generate a sanitized filename
    file_name = f"{sanitize_filename(model_name)}_{sanitize_filename(language)}_evaluation.json"
    file_path = os.path.join(model_dir, file_name)

    # Save results as a JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

# Load already evaluated results from a file if available
def load_evaluated_results(output_directory, model_name, language):
    model_dir = os.path.join(output_directory, sanitize_filename(model_name), sanitize_filename(language))
    file_name = f"{sanitize_filename(model_name)}_{sanitize_filename(language)}_evaluation.json"
    file_path = os.path.join(model_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        return {}

# Perform evaluation of each scenario's empathetic response using GPT model
def evaluate_scenario(conv_id, dialogue, scenario_name, empathetic_response, criteria, language):
    result = {
        "scenario": scenario_name,
        "final_empathetic_statement": empathetic_response,
        "evaluations": {},
        "scores": {}
    }

    # General prompt shared across evaluations
    common_prompt = """
    You will be given one response for one dialogue.

    Your task is to rate the response based on the criteria provided.

    Please make sure you read and understand these criteria carefully. Refer back to them as needed during your evaluation.
    """

    # Detailed descriptions of each evaluation criterion
    criteria_prompts = {
        "Explorations (EX)": """
        **Explorations (EX)** (1-5) - This criterion evaluates whether the response shows active interest in further exploring the interlocutor's situation or feelings and attempts to probe emotions or experiences that have not been explicitly stated.
        Score 1: No attempt to explore the interlocutor's emotions or experiences, showing no additional inquiry or interest.
        Score 2: General attempts at exploration are made, but they are not specific or fail to delve deeply into the interlocutor's situation.
        Score 3: Somewhat specific exploration is attempted, but it lacks depth or completeness.
        Score 4: The response makes a fairly specific and clear attempt to explore the interlocutor's feelings or experiences.
        Score 5: The response makes a very specific and thoughtful attempt to understand the interlocutor's emotions and experiences deeply.""",

        "Interpretations (IP)": """
        **Interpretations (IP)** (1-5) - This criterion assesses whether the response recognizes and interprets the interlocutor's emotions and experiences, showing an understanding of what the interlocutor is feeling.
        Score 1: The response does not show any acknowledgment or interpretation of the interlocutor's feelings or experiences.
        Score 2: The response acknowledges the interlocutor’s situation or feelings but does so at a very surface level.
        Score 3: The response shows some understanding of the interlocutor’s feelings or experiences, but it lacks depth.
        Score 4: The response provides a fairly deep interpretation of the interlocutor’s emotions and experiences, showing considerable understanding.
        Score 5: The response offers a very deep and clear interpretation, fully conveying an understanding of the interlocutor’s feelings and experiences.""",
        
        "Emotional Reactions (ER)": """
        **Emotional Reactions (ER)** (1-5) - This criterion evaluates whether the response expresses emotional reactions such as warmth, compassion, and concern towards the interlocutor’s situation.
        Score 1: The response lacks any expression of emotional reactions, warmth, compassion, or concern.
        Score 2: Emotional reactions are present but are either unclear or insufficiently warm or compassionate.
        Score 3: The response shows some emotional reaction but lacks sufficient depth or warmth.
        Score 4: The response expresses a significant level of emotional reaction, showing warmth, concern, and deep compassion for the interlocutor's situation.
        Score 5: The response shows a very deep and warm emotional reaction, displaying strong empathy and concern for the interlocutor's situation.""",

        "Evoked Emotion Alignment (EEA)": """
        **Evoked Emotion Alignment (EEA)** (1-5) - This criterion evaluates how well the LLM aligns with or influences the emotional response of a human when presented with a specific situation.
        Score 1: The response fails to recognize the interlocutor's emotional state and may respond with an inappropriate emotional tone, potentially exacerbating negative feelings.
        Score 2: The response minimally recognizes the interlocutor's emotional state but reacts insufficiently, resulting in little to no noticeable change in the interlocutor's emotions.
        Score 3: The response acknowledges the emotional state and partially addresses it, but falls short of a fully appropriate reaction, leading to a moderate change in the interlocutor's emotional state.
        Score 4: The response accurately recognizes the emotional state and effectively addresses it, alleviating some of the negative feelings or appropriately balancing overly excited emotions.
        Score 5: The response demonstrates highly accurate recognition of the emotional state, perfectly aligns its response, and effectively modulates the emotional state, providing clear emotional relief or balance when needed.""",

        "Cultural Appropriateness (CA)": f"""
        **Cultural Appropriateness (CA)** (1-5) - This criterion evaluates how well the response functions within the context of {language} culture, including traditions, values, customs, social expectations, and grammatical correctness in {language}.
        Score 1: The response is very disconnected from {language} culture and contains significant grammatical errors, leading to potential cultural misunderstandings.
        Score 2: The response somewhat misaligns with {language} culture and has some grammatical issues, with insufficient reflection of cultural context and expression.
        Score 3: The response generally aligns with {language} culture and is grammatically correct with no major issues, but some expressions or grammatical usage may feel slightly awkward or incomplete.
        Score 4: The response mostly aligns with {language} culture and is grammatically appropriate, reflecting cultural context and expression well.
        Score 5: The response perfectly aligns with {language} culture and is grammatically accurate and natural, fully reflecting traditions, customs, and social expectations."""

    }

    # Loop through each criterion to evaluate the response
    for criterion in criteria:
        system_prompt = common_prompt + criteria_prompts[criterion]

        # Create the user prompt including the dialogue and empathetic response
        user_prompt = f"""
        **Dialogue:**
        {dialogue}

        **Empathetic Response:**
        {empathetic_response}

        Please give feedback on the listener’s responses. Also, provide the listener with a score on a scale of 1 to 5 for the **{criterion}**, where a higher score indicates better overall performance. Make sure to give feedback or comments for the **{criterion}** first and then write the score for the **{criterion}**.

        **Response Format:**
        Feedback: [Your feedback here]
        Score: [1-5]"""

        # Retry mechanism to handle potential API errors
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()}
                    ],
                    temperature=0.7,  
                    max_tokens=256  
                )

                # Extract and process the GPT response
                assistant_content = response['choices'][0]['message']['content'].strip()
                feedback, score = assistant_content.split("Score:")  
                score = int(score.strip()) 
                result['evaluations'][criterion] = feedback.strip()  
                result['scores'][criterion] = score  

                break  

            except Exception as e:
                print(f"Error evaluating {criterion} for conv_id {conv_id}, scenario {scenario_name}: {e}")
                time.sleep(1)  
                if attempt == max_retries - 1:
                    result['evaluations'][criterion] = f"Error: {e}"
                    result['scores'][criterion] = "Error"

    return result

# Main function to execute the evaluation process
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    base_directory = os.path.join(project_root, 'output', 'experiment_results', 'sample')
    output_directory = os.path.join(project_root, 'output', 'eval_results', 'sample')

    models = [
        "claude-3-5-sonnet-20240620",
        "Meta-Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Qwen2-7B-Instruct",
        "EXAONE-3.0-7.8B-Instruct"
    ]
    languages = ["Korean","English"]

    # List of evaluation criteria
    criteria = [
        "Explorations (EX)",
        "Interpretations (IP)",
        "Emotional Reactions (ER)",
        "Evoked Emotion Alignment (EEA)",
        "Cultural Appropriateness (CA)"
    ]

    # Iterate over each model and language combination
    for model_name in models:
        for language in languages:
            input_file = os.path.join(base_directory, f"results_{model_name}_{language}.json")
            data = load_json_data(input_file)  

            # Check if the input file exists
            if not os.path.exists(input_file):
                print(f"Input file not found: {input_file}")
                continue

            # Load previously evaluated results (if any)
            results = load_evaluated_results(output_directory, model_name, language)

            print(f"Processing model: {model_name}, language: {language}")

            # Iterate through the entries in the JSON data
            for entry in tqdm(data.values(), desc=f"{model_name} - {language}"):
                conv_id = entry.get("conv_id")  
                dialogue = entry.get("dialogue", "") 
                scenarios = entry.get("scenarios", [])  

                for scenario in scenarios:
                    scenario_name = scenario.get("scenario")  
                    empathetic_response = scenario.get("final_empathetic_statement")  

                    # Skip evaluation if this scenario has already been evaluated
                    if conv_id in results and scenario_name in results[conv_id]:
                        print(f"Skipping already evaluated scenario: {scenario_name} for conv_id {conv_id}")
                        continue

                    # Perform the evaluation for each scenario
                    evaluation_result = evaluate_scenario(
                        conv_id=conv_id,
                        dialogue=dialogue,
                        scenario_name=scenario_name,
                        empathetic_response=empathetic_response,
                        criteria=criteria,
                        language=language
                    )

                    # Update results with the new evaluation and save to file
                    results = {**results, **{conv_id: {**results.get(conv_id, {}), **{scenario_name: evaluation_result}}}}
                    save_evaluation_to_file(
                        results=results,
                        output_directory=output_directory,
                        model_name=model_name,
                        language=language
                    )

# Run the main function when the script is executed
if __name__ == "__main__":
    main()