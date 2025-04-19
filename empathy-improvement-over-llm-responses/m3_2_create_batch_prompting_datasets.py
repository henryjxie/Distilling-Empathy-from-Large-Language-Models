import pandas
import json

dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

# Naive prompt
prompts = []
for i, row in dataset.iterrows():
    prompts.append({
        "custom_id": f"response_{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content":
                        "Below is a response to a given speaker utterance in a given context. "
                        "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. \n"
                        + f"Context: {row['situation']} \n"
                        + f"Speaker Utterance: {row['speaker_uttr']} \n"
                        + f"Response: {row['initial_response']} \n"
                        + "Improved Response:"
                }
            ]
        }
    })

with open("../datasets/llm-impr/naive_prompt_improve_initial_response.jsonl", "w", encoding="utf-8") as file:
    for item in prompts:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# Instruct to improve on all three dimensions of empathy
prompts = []
for i, row in dataset.iterrows():
    prompts.append({
        "custom_id": f"response_{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content":
                        "Below is a response to a given speaker utterance in a given context. " + \
                        "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                        "Your higher quality response should be improved along the three dimensions of empathy: cognitive, affective (emotional), and compassionate empathy. " + \
                        "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                        "It is being able to see the world through their eyes and understand their point of view. " + \
                        "Affective empathy is the ability to experience the emotions of another person. " + \
                        "It is feeling what they are feeling, both positive and negative. " + \
                        "Compassionate empathy is the ability to not only understand and share another person's feelings, but also to be moved to help if needed. " + \
                        "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another's distress or suffering. \n"
                        + f"Context: {row['situation']} \n"
                        + f"Speaker Utterance: {row['speaker_uttr']} \n"
                        + f"Response: {row['initial_response']} \n"
                        + "Improved Response:"
                }
            ]
        }
    })

with open("../datasets/llm-impr/improve_all_three_dimensions_improve_initial_response.jsonl", "w", encoding="utf-8") as file:
    for item in prompts:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# Instruct to identify which dimensions of empathy the original response is lacking the most in, and improve along that dimension of empathy
prompts = []
for i, row in dataset.iterrows():
    prompts.append({
        "custom_id": f"response_{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content":
                        "Below is a response to a given speaker utterance in a given context. " + \
                        "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                        "In the process of generating a higher quality empathetic response, you should identify the dimension of empathy (cognitive, affective, and compassionate dimensions) that the original response lacks most of, and specifically improve along the lines of the dimension you identified. " + \
                        "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                        "It is being able to see the world through their eyes and understand their point of view. " + \
                        "Affective empathy is the ability to experience the emotions of another person. " + \
                        "It is feeling what they are feeling, both positive and negative. " + \
                        "Compassionate empathy is the ability to not only understand and share another person's feelings, but also to be moved to help if needed. " + \
                        "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another's distress or suffering. \n"
                        + f"Context: {row['situation']} \n"
                        + f"Speaker Utterance: {row['speaker_uttr']} \n"
                        + f"Response: {row['initial_response']} \n"
                        + "Improved Response:"
                }
            ]
        }
    })

with open("../datasets/llm-impr/identify_lacking_dimension_improve_initial_response.jsonl", "w", encoding="utf-8") as file:
    for item in prompts:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# Instruct to improve on one of the three dimensions only (cognitive)
prompts = []
for i, row in dataset.iterrows():
    prompts.append({
        "custom_id": f"response_{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content":
                        "Below is a response to a given speaker utterance in a given context. " + \
                        "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                        "Your higher quality response should be improved specifically along the cognitive dimension of empathy. " + \
                        "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                        "It is being able to see the world through their eyes and understand their point of view. \n"
                        + f"Context: {row['situation']} \n"
                        + f"Speaker Utterance: {row['speaker_uttr']} \n"
                        + f"Response: {row['initial_response']} \n"
                        + "Improved Response:"
                }
            ]
        }
    })

with open("../datasets/llm-impr/improve_cognitive_improve_initial_response.jsonl", "w", encoding="utf-8") as file:
    for item in prompts:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# Instruct to improve on one of the three dimensions only (affective)
prompts = []
for i, row in dataset.iterrows():
    prompts.append({
        "custom_id": f"response_{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content":
                        "Below is a response to a given speaker utterance in a given context. " + \
                        "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                        "Your higher quality response should be improved specifically along the affective (emotional) dimension of empathy. " + \
                        "Affective empathy is the ability to experience the emotions of another person. " + \
                        "It is feeling what they are feeling, both positive and negative. \n"
                        + f"Context: {row['situation']} \n"
                        + f"Speaker Utterance: {row['speaker_uttr']} \n"
                        + f"Response: {row['initial_response']} \n"
                        + "Improved Response:"
                }
            ]
        }
    })

with open("../datasets/llm-impr/improve_affective_improve_initial_response.jsonl", "w", encoding="utf-8") as file:
    for item in prompts:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

# Instruct to improve on one of the three dimensions only (compassionate)
prompts = []
for i, row in dataset.iterrows():
    prompts.append({
        "custom_id": f"response_{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content":
                        "Below is a response to a given speaker utterance in a given context. " + \
                        "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                        "Your higher quality response should be improved specifically along the compassionate dimension of empathy. " + \
                        "Compassionate empathy is the ability to not only understand and share another person's feelings, but also to be moved to help if needed. " + \
                        "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another's distress or suffering. \n"
                        + f"Context: {row['situation']} \n"
                        + f"Speaker Utterance: {row['speaker_uttr']} \n"
                        + f"Response: {row['initial_response']} \n"
                        + "Improved Response:"
                }
            ]
        }
    })

with open("../datasets/llm-impr/improve_compassionate_improve_initial_response.jsonl", "w", encoding="utf-8") as file:
    for item in prompts:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")