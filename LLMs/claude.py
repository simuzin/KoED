import os
import json
import anthropic
from tqdm import tqdm

# Set Claude API key (replace 'YOUR_ANTHROPIC_API_KEY_HERE' with the actual API key)
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"  # Placeholder for the API key
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Get the current script's directory and build a relative path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, '..', 'data', 'KoED_sample_100.json')

# Load dialogue data from the specified JSON file
with open(data_file_path, 'r', encoding='utf-8') as f:
    dialogues_data = json.load(f)

# Function to generate a response using the Claude API
def get_response_from_claude(prompt, system_instruction):
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=256,  
            temperature=0.1,  
            system=system_instruction,  
            messages=[{"role": "user", "content": prompt}]  
        )
        return response
    except Exception as e:
        return {"error": str(e)}

# Function to extract text from a list of TextBlock objects
def extract_text_blocks(content):
    return "\n".join(block.text for block in content)

# Generate empathetic dialogues in (KoED & ED)
for lang, lang_key in [("Korean", "ko_utter"), ("English", "utter")]:
    # Build the output file path relative to the current script
    output_file = os.path.join(current_dir, '..', 'output', 'experiment_results', 'sample', f'results_claude-3-5-sonnet-20240620_{lang}.json')
    
    # Load existing results if available; if not, initialize as an empty dictionary
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            outputs_summary = json.load(f)
    except FileNotFoundError:
        outputs_summary = {}  

    common_task_definition = f"""Task Definition: This is a/an {lang.lower()} empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others' feelings in a conversation as much as possible. Guideline Instruction: Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener."""

    # Iterate over each dialogue
    for index, dialogue_data in enumerate(tqdm(dialogues_data, desc=f"Processing {lang} Dialogues for Claude")):
        conv_id = dialogue_data['conv_id'] 

        # Skip previously processed dialogues or specific ones (re-experiment parts)
        if conv_id in outputs_summary:
            continue

        # Initialize dialogue text with alternating turns (Speaker/Listener)
        dialogue_text = ""
        speaker_turn = True  # Assume Speaker starts first

        for utterance in dialogue_data['dialogue']:
            if lang_key in utterance and utterance[lang_key]: 
                if speaker_turn:
                    dialogue_text += f"Speaker: {utterance[lang_key]}\n"
                else:
                    dialogue_text += f"Listener: {utterance[lang_key]}\n"
                speaker_turn = not speaker_turn 

        # If the number of utterances is even, remove the last one for balance
        num_utterances = dialogue_text.count('\n')
        if num_utterances % 2 == 0:
            dialogue_text = '\n'.join(dialogue_text.split('\n')[:-2]) + '\n'

        # Format the multi-turn dialogue
        multi_turn_dialogue = f"\n    Multi-Turn Dialogue:\n        {dialogue_text}"

        ## Scenario generation
        scenarios = [
            ("34개의 단일 감정", [ # 34-Single
                common_task_definition + """
                List of 34 Emotions:
                    Afraid (두려움), Angry (화남), Annoyed (짜증남), Anticipating (기대됨), Anxious (불안함), Apprehensive (염려됨), Ashamed (부끄러움), Caring (보살핌), Confident (자신감), Content (만족함), Devastated (충격받음), Disappointed (실망함), Disgusted (역겨움), Embarrassed (당황함), Excited (흥분됨), Faithful (충실함), Furious (격노함), Grateful (감사함), Guilty (죄책감), Hopeful (희망적), Impressed (감명받음), Jealous (질투남), Joyful (기쁨), Lonely (외로움), Nostalgic (향수에 젖음), Prepared (준비됨), Proud (자랑스러움), Sad (슬픔), Sentimental (감상적), Surprised (놀람), Terrified (겁에 질림), Trusting (신뢰함), 정 (a feeling that arises in one's heart, or a feeling of love or affinity), 한 (a feeling of bitter resentment, injustice, pity, or sadness)

                Important Guidelines:
                    - Do not use any emotion terms other than the 34 basic emotions listed above.
                    - Combinations or mixtures of emotions are not allowed. Choose and use only one emotion.
                    - Even for complex or subtle emotions, you must express them using only one of the 34 emotions that is closest in meaning.""",
                f"""
                    {multi_turn_dialogue}

                Step-by-Step Instructions:
                    1. Analyze the given dialogue to identify the Speaker's emotional state.
                    2. Specify the identified emotion using only one of the 34 basic emotions listed above.
                    (STOP HERE. Do NOT proceed to steps 3 and 4 yet. Only identify the emotion at this stage.)
                """,
            ]),
            ("34개의 멀티 감정", [ # 34-Multi
                common_task_definition + """
                List of 34 Emotions:
                    Afraid (두려움), Angry (화남), Annoyed (짜증남), Anticipating (기대됨), Anxious (불안함), Apprehensive (염려됨), Ashamed (부끄러움), Caring (보살핌), Confident (자신감), Content (만족함), Devastated (충격받음), Disappointed (실망함), Disgusted (역겨움), Embarrassed (당황함), Excited (흥분됨), Faithful (충실함), Furious (격노함), Grateful (감사함), Guilty (죄책감), Hopeful (희망적), Impressed (감명받음), Jealous (질투남), Joyful (기쁨), Lonely (외로움), Nostalgic (향수에 젖음), Prepared (준비됨), Proud (자랑스러움), Sad (슬픔), Sentimental (감상적), Surprised (놀람), Terrified (겁에 질림), Trusting (신뢰함), 정 (a feeling that arises in one's heart, or a feeling of love or affinity), 한 (a feeling of bitter resentment, injustice, pity, or sadness)

                Important Guidelines:
                    - Do not use any emotion terms other than the 34 basic emotions listed above.
                    - Combinations or mixtures of emotions are allowed. Select up to 4 emotions that best describe the Speaker's emotional state.""",
                f"""
                {multi_turn_dialogue}

                Step-by-Step Instructions:
                    1. Analyze the given dialogue to identify the Speaker's complex emotional state.
                    2. Specify the identified emotions using multiple labels from the 34 emotions listed above. Select all that apply, with no minimum or maximum limit.
                    (STOP HERE. Do NOT proceed to steps 3 and 4 yet. Only identify the emotion at this stage.)
                """
            ]),
        ]

        # Dictionary to store dialogue results
        dialogue_results = {
            "conv_id": conv_id,
            "dialogue": dialogue_text,
            "scenarios": []
        }

        scenario_next_steps = {
            "34개의 단일 감정": 3,
            "34개의 멀티 감정": 3
        }

        # Process each scenario for the current dialogue
        for scenario_name, scenario_content in scenarios:
            # Step 1: Identify emotions
            if scenario_name in ["34개의 단일 감정", "34개의 멀티 감정"]:
                identified_emotions_response = get_response_from_claude(scenario_content[1], system_instruction=scenario_content[0])
                identified_emotions_text = extract_text_blocks(identified_emotions_response.content)
                identified_emotions = {"role": "assistant", "content": identified_emotions_text}
            else:
                identified_emotions = None

            # Step 2: Generate empathetic response
            if identified_emotions:
                scenario_content[1] += f"\n\nIdentified Emotions: {identified_emotions['content']}\n\n{scenario_next_steps[scenario_name]}. Proceeding with the next {lang} empathetic response based on the identified emotions."
               
            empathetic_response = get_response_from_claude(scenario_content[1], system_instruction=scenario_content[0])
            empathetic_response_text = extract_text_blocks(empathetic_response.content)
            empathetic_response_content = {"role": "assistant", "content": f"Listener: {empathetic_response_text}"}

            # Append scenario results to dialogue results
            dialogue_results["scenarios"].append({
                "scenario": scenario_name,
                "identified_emotions": identified_emotions,
                "empathetic_response": empathetic_response_content
            })

        # Add the current dialogue results to the overall output
        outputs_summary[conv_id] = dialogue_results

        # Save the results to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs_summary, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file} successfully.")
