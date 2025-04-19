from openai import OpenAI
import pandas as pd
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

rlhfDataset = pd.read_csv("../datasets/generate_initial_response.csv")
context = rlhfDataset["situation"].to_list()
speaker_utterance = rlhfDataset["speaker_uttr"].to_list()
response = rlhfDataset["initial_response"].to_list()

def getResponse(context, speaker_utterance, response):
    prompt = [{"role": "user", "content":
                "Below is a response to a given speaker utterance in a given context. " + \
                "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                "Your higher quality response should be improved specifically along the cognitive dimension of empathy. " + \
                "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                "It is being able to see the world through their eyes and understand their point of view. \n"
                f"Context: {context} \n" + \
                f"Speaker Utterance: {speaker_utterance} \n" + \
                f"Response: {response} \n" + \
                "Improved Response:"}]

    llmResponse = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt
    )

    llm_response_cognitive = llmResponse.choices[0].message.content

    prompt = [{"role": "user", "content":
                "Below is a response to a given speaker utterance in a given context. " + \
                "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                "Your higher quality response should be improved specifically along the cognitive dimension of empathy. " + \
                "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                "It is being able to see the world through their eyes and understand their point of view. \n"
                f"Context: {context} \n" + \
                f"Speaker Utterance: {speaker_utterance} \n" + \
                f"Response: {response} \n" + \
                "Improved Response:"},
                {"role": "assistant", "content": f"{llm_response_cognitive}"},
                {"role": "user", "content":
                "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                "Your higher quality response should be improved specifically along the affective (emotional) dimension of empathy. " + \
                "Affective empathy is the ability to experience the emotions of another person. " + \
                "It is feeling what they are feeling, both positive and negative. \n" + \
                "Improved Response:"}]

    llmResponse = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt
    )

    llm_response_affective = llmResponse.choices[0].message.content

    prompt = [{"role": "user", "content":
                "Below is a response to a given speaker utterance in a given context. " + \
                "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                "Your higher quality response should be improved specifically along the cognitive dimension of empathy. " + \
                "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                "It is being able to see the world through their eyes and understand their point of view. \n"
                f"Context: {context} \n" + \
                f"Speaker Utterance: {speaker_utterance} \n" + \
                f"Response: {response} \n" + \
                "Improved Response:"},
                {"role": "assistant", "content": f"{llm_response_cognitive}"},
                {"role": "user", "content":
                "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                "Your higher quality response should be improved specifically along the affective (emotional) dimension of empathy. " + \
                "Affective empathy is the ability to experience the emotions of another person. " + \
                "It is feeling what they are feeling, both positive and negative. \n" + \
                "Improved Response:"},
                {"role": "assistant", "content": f"{llm_response_affective}"},
                {"role": "user", "content":
                "Generate a new improved empathetic response, using on average 28 words and a maximum of 97 words, that is of higher empathetic quality and also retains the original meaning, intention, and emotion of the original response. " + \
                "Your higher quality response should be improved specifically along the compassionate dimension of empathy. " + \
                "Compassionate empathy is the ability to not only understand and share another person's feelings, but also to be moved to help if needed. " + \
                "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another's distress or suffering. \n" + \
                "Improved Response:"}]

    llmResponse = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt
    )

    return llmResponse.choices[0].message.content

for i in range(len(rlhfDataset)):
    print(i + 1)
    get_response_output = getResponse(context[i], speaker_utterance[i], response[i])
    print(get_response_output)
    print()

    rlhfDataset.loc[i, "improved_initial_response"] = get_response_output
    
    rlhfDataset.to_csv("../datasets/llm-impr/improve_three_dimensions_sequentially_improve_initial_response.csv", index=False)