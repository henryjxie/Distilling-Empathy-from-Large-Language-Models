import json
import pandas as pd

def csvToJson(context, speakerUtterance, response, improvedResponse, prompts):
    
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
                    
                        {"role": "user", "content": "The response generated is a low-quality empathetic response. " + \
                                "Generate a new response, using on average 28 words and a maximum of 97 words, that is of higher quality and also retains the original meaning, intention, and emotion of the original response. " + \
                                "A high quality response is one that exhibits strong empathy, is of high textual quality, and demonstrates clear expression and rich emotions."},
                        
                        {"role": "assistant", "content": improvedResponse}
            ]
        }
    )
    return prompts


def dataset_to_json_model_rlhf(csv_path, json_path): 
    rlhfDataset = pd.read_csv(csv_path)
    context = rlhfDataset["situation"].to_list()
    speakerUtterance = rlhfDataset["speaker_uttr"].to_list()
    human_response = rlhfDataset["response_human"].to_list()
    model_response = rlhfDataset["response_model"].to_list()

    jsonFilePath = json_path

    prompts = []
    for i in range(len(context)):
        prompts = csvToJson(context[i], speakerUtterance[i], human_response[i], model_response[i], prompts)

    with open(jsonFilePath, mode="w", encoding="utf-8") as jsonFile:
        json.dump(prompts, jsonFile, indent=4)