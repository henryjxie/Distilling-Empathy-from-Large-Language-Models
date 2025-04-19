import os
import pandas
import json

from openai import OpenAI

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def submit_batch(): 

    dataset = pandas.read_csv("../datasets/dataset.csv")

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
                            "Empathy is the ability to understand and share the feelings of another person. "
                            + "It is the ability to put yourself in someone else's shoes and see the world from their perspective. "
                            + "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. "
                            + "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. "
                            + "It is being able to see the world through their eyes and understand their point of view. "
                            + "Affective empathy is the ability to experience the emotions of another person. "
                            + "It is feeling what they are feeling, both positive and negative. "
                            + "Compassionate empathy is the ability to not only understand and share another person's feelings, but also to be moved to help if needed. "
                            + "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another's distress or suffering. "
                            + "Empathy is important because it allows us to connect with others on a deeper level. "
                            + "It helps us to build trust, compassion, and intimacy. "
                            + "Empathy is also essential for effective communication and conflict resolution. "
                            + "You are engaging in a conversation with a human. "
                            + "Respond in an empathetic manner to the following speaker utterance in a given context using on average 28 words and a maximum of 97 words.\n"
                            + f"Context: {row['situation']} \n"
                            + f"Speaker Utterance: {row['speaker_uttr']} \n"
                            + "Response:"
                    }
                ]
            }
        })

    with open("../datasets/llm-impr/generate_initial_response.jsonl", "w", encoding="utf-8") as file:
        for item in prompts:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Naive prompt - SFT
    batch_input_file = client.files.create(
        file=open("../datasets/llm-impr/generate_initial_response.jsonl", "rb"),
        purpose="batch"
    )

    batch_object = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Generate initial response"
        }
    )

    batch_object = batch_object.to_dict()

    with open("../datasets/llm-impr/generate_intial_response_batch_object.json", "w", encoding="utf-8") as file:
        json.dump(batch_object, file, indent=2)


def get_results(): 
    with open("../datasets/llm-impr/generate_intial_response_batch_object.json") as file:
        data = json.load(file)
        print(client.batches.retrieve(data["id"]).to_dict()["status"])
        client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

        with open("../datasets/llm-impr/generate_initial_response_batch_results.jsonl", "wb") as file:
            file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)


def add_to_original(): 
    original_dataset = pandas.read_csv("../datasets/dataset.csv")

    data = []
    with open("../datasets/llm-impr/generate_initial_response_batch_results.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            response = json.loads(line)
            response = response["response"]["body"]["choices"][0]["message"]["content"]
            data.append(response)
        original_dataset["initial_response"] = data
        original_dataset.to_csv("../datasets/llm-impr/generate_initial_response.csv", index=False)

def menu():
    print("=== Main Menu ===")
    print("1. Submit Batch")
    print("2. Get Results")
    print("3. Add to Original Dataset")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        submit_batch()
    elif choice == '2':
        get_results()
    elif choice == '3':
        add_to_original()
    elif choice == '4':
        return False
    else:
        print("Invalid choice. Try again.")

    return True

# Loop until user exits
while True:
    if not menu():
        break