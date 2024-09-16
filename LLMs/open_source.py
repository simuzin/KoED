import os
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm



# List of model IDs to use for generating responses
model_ids = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

# Load dialogue data from the JSON file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, '..', 'data', 'KoED_sample_100.json')
with open(data_file_path, 'r', encoding='utf-8') as f:
    dialogues_data = json.load(f)


# Iterate through each model
for model_id in model_ids:

    # Set up quantization configuration for 4-bit loading
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
    )

    # Load the model and tokenizer for the current model ID
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        cache_dir="/data",
        device_map="auto",
        trust_remote_code=True  
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/data")
    
    # Create a text generation pipeline for each model
    text_generation_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

   
    # Generate empathetic dialogues in (KoED & ED)
    for lang, lang_key in [("Korean", "ko_utter"), ("English", "utter")]:
        model_name = model_id.split("/")[-1]
        output_file = os.path.join(current_dir, '..', 'output', 'eval_results', 'sample', f'results_{model_name}_{lang}.json')

        # Load existing results if available
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                outputs_summary = json.load(f)
        except FileNotFoundError:
            outputs_summary = {}

        # Define the common task instruction for the model to follow
        common_task_definition = f"""Task Definition: This is a/an {lang.lower()} empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others' feelings in a conversation as much as possible. Guideline Instruction: Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener."""

        # Process each dialogue in the JSON file (using tqdm for progress tracking)
        for index, dialogue_data in enumerate(tqdm(dialogues_data, desc=f"Processing {lang} Dialogues for {model_name}")):
            conv_id = dialogue_data['conv_id']

            # Skip previously processed dialogues or specific ones (re-experiment parts)
            if conv_id in outputs_summary:
                continue

            # Initialize the dialogue text with alternating turns between Speaker and Listener
            dialogue_text = ""
            speaker_turn = True 

            for utterance in dialogue_data['dialogue']:
                if lang_key in utterance and utterance[lang_key]:  
                    if speaker_turn:
                        dialogue_text += f"Speaker: {utterance[lang_key]}\n"
                    else:
                        dialogue_text += f"Listener: {utterance[lang_key]}\n"
                    speaker_turn = not speaker_turn  

            # Count the number of utterances
            num_utterances = dialogue_text.count('\n')

            # If even number of utterances, remove the last one for balance
            if num_utterances % 2 == 0:
                dialogue_text = '\n'.join(dialogue_text.split('\n')[:-2]) + '\n'

            multi_turn_dialogue = f"\n    Multi-Turn Dialogue:\n        {dialogue_text}"

            ## Scenario generation
            scenarios = [
                ("34개의 단일 감정", [ # 34-Single
                    {"role": "system", "content": common_task_definition + """
                        List of 34 Emotions:
                            Afraid (두려움), Angry (화남), Annoyed (짜증남), Anticipating (기대됨), Anxious (불안함), Apprehensive (염려됨), Ashamed (부끄러움), Caring (보살핌), Confident (자신감), Content (만족함), Devastated (충격받음), Disappointed (실망함), Disgusted (역겨움), Embarrassed (당황함), Excited (흥분됨), Faithful (충실함), Furious (격노함), Grateful (감사함), Guilty (죄책감), Hopeful (희망적), Impressed (감명받음), Jealous (질투남), Joyful (기쁨), Lonely (외로움), Nostalgic (향수에 젖음), Prepared (준비됨), Proud (자랑스러움), Sad (슬픔), Sentimental (감상적), Surprised (놀람), Terrified (겁에 질림), Trusting (신뢰함), 정 (a feeling that arises in one's heart, or a feeling of love or affinity), 한 (a feeling of bitter resentment, injustice, pity, or sadness)

                        Important Guidelines:
                            - Do not use any emotion terms other than the 34 basic emotions listed above.
                            - Combinations or mixtures of emotions are not allowed. Choose and use only one emotion.
                            - Even for complex or subtle emotions, you must express them using only one of the 34 emotions that is closest in meaning."""},
                    {"role": "user", "content": f"""
                        {multi_turn_dialogue}

                    Step-by-Step Instructions:
                        1. Analyze the given dialogue to identify the Speaker's emotional state.
                        2. Specify the identified emotion using only one of the 34 basic emotions listed above.
                        (STOP HERE. Do NOT proceed to steps 3 and 4 yet. Only identify the emotion at this stage.)
                    """},
                ]),
                ("34개의 멀티 감정", [ # 34-Multi
                    {"role": "system", "content": common_task_definition + """
                        List of 34 Emotions:
                            Afraid (두려움), Angry (화남), Annoyed (짜증남), Anticipating (기대됨), Anxious (불안함), Apprehensive (염려됨), Ashamed (부끄러움), Caring (보살핌), Confident (자신감), Content (만족함), Devastated (충격받음), Disappointed (실망함), Disgusted (역겨움), Embarrassed (당황함), Excited (흥분됨), Faithful (충실함), Furious (격노함), Grateful (감사함), Guilty (죄책감), Hopeful (희망적), Impressed (감명받음), Jealous (질투남), Joyful (기쁨), Lonely (외로움), Nostalgic (향수에 젖음), Prepared (준비됨), Proud (자랑스러움), Sad (슬픔), Sentimental (감상적), Surprised (놀람), Terrified (겁에 질림), Trusting (신뢰함), 정 (a feeling that arises in one's heart, or a feeling of love or affinity), 한 (a feeling of bitter resentment, injustice, pity, or sadness)

                        Important Guidelines:
                            - Combinations or mixtures of emotions are allowed.
                            - Select up to 4 emotions that best describe the Speaker's emotional state."""},
                    {"role": "user", "content": f"""
                        {multi_turn_dialogue}

                    Step-by-Step Instructions:
                        1. Analyze the given dialogue to identify the Speaker's complex emotional state.
                        2. Specify the identified emotions using multiple labels from the 34 emotions listed above. Select all that apply, with no minimum or maximum limit.
                        (STOP HERE. Do NOT proceed to steps 3 and 4 yet. Only identify the emotion at this stage.)
                    """},
                ]),
            ]

            # Initialize a dictionary to store dialogue results
            dialogue_results = {
                "conv_id": conv_id,
                "dialogue": dialogue_text,
                "scenarios": []
            }

            scenario_next_steps = {
                "34개의 단일 감정" : 3,
                "34개의 멀티 감정": 3
            }

            # Process each scenario
            for scenario_name, scenario in scenarios:
                # Step 1: Identify emotions using the model
                if scenario_name in ["34개의 단일 감정", "34개의 멀티 감정"]:
                    emotion_output = text_generation_pipeline(
                        scenario,
                        max_new_tokens=256,
                    )
                    identified_emotions = emotion_output[0]['generated_text'][-1]
                else:
                    identified_emotions = None
                
                # Step 2: Generate empathetic response based on identified emotions
                if identified_emotions:
                    scenario[1]["content"] += f"\n\nIdentified Emotions: {identified_emotions}\n\n{scenario_next_steps[scenario_name]}. Generate the next {lang} empathetic response based on the identified emotions."

                empathetic_response_output = text_generation_pipeline(
                    scenario,
                    max_new_tokens=256,
                )
                empathetic_response = empathetic_response_output[0]['generated_text'][-1]
                
                # Append scenario results to dialogue results
                dialogue_results["scenarios"].append({
                    "scenario": scenario_name,
                    "identified_emotions": identified_emotions,
                    "empathetic_response": f"Listener: {empathetic_response}"
                })

            # Add the current dialogue results to the summary
            outputs_summary[conv_id] = dialogue_results

            # Save the results immediately to avoid data loss
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(outputs_summary, f, ensure_ascii=False, indent=4)

        # Notify the user of successful save
        print(f"Results saved to {output_file} successfully.")
