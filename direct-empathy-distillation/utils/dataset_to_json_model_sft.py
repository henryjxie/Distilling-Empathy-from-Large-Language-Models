import json
import pandas as pd

def csvToJson(context, speakerUtterance, response, prompts):
    
    prompts.append(
        {
            "messages":[
                        {"role": "user", "content": "Empathy is the ability to understand and share the feelings of another person. " + \
                            "It is the ability to put yourself in someone else’s shoes and see the world from their perspective. " + \
                            "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. " + \
                            "Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. " + \
                            "It is being able to see the world through their eyes and understand their point of view. " + \
                            "Affective empathy is the ability to experience the emotions of another person. " + \
                            "It is feeling what they are feeling, both positive and negative. " + \
                            "Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed. " + \
                            "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another’s distress or suffering. " + \
                            "Empathy is important because it allows us to connect with others on a deeper level. " + \
                            "It helps us to build trust, compassion, and intimacy. Empathy is also essential for effective communication and conflict resolution. " + \
                            "You are engaging in a conversation with a human. Respond in an empathetic manner to the following speaker utterance in a given context using on average 28 words and a maximum of 97 words. \n" + \
                            f"Context: {context} \n" + \
                            f"Speaker Utterance: {speakerUtterance} \n" + \
                            "Response: "},

                        {"role": "assistant", "content": response},
            ]
        }
    )
    return prompts


def dataset_to_json_model_sft(csv_path, json_path): 
    sftDataset = pd.read_csv(csv_path)
    context = sftDataset["situation"].to_list()
    speakerUtterance = sftDataset["speaker_uttr"].to_list()
    human_response = sftDataset["response_human"].to_list()
    model_response = sftDataset["response_model"].to_list()

    jsonFilePath = json_path

    prompts = []

    for i in range(len(sftDataset)):
        for j in range(0, 2):
            if j == 0:
                prompts = csvToJson(context[i], speakerUtterance[i], human_response[i], prompts)
            else:
                prompts = csvToJson(context[i], speakerUtterance[i], model_response[i], prompts)

    with open(jsonFilePath, mode="w", encoding="utf-8") as jsonFile:
        json.dump(prompts, jsonFile, indent=4)