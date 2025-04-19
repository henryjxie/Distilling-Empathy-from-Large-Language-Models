import json

def convert_rlhf_to_dpo(rlhf_file_path, dpo_file_path):
    """
    Converts RLHF JSON format to DPO JSON format.

    Args:
        rlhf_file_path (str): Path to the RLHF JSON file.
        dpo_file_path (str): Path to save the transformed DPO JSON file.
    """
    # Load the RLHF data
    with open(rlhf_file_path, 'r') as rlhf_file:
        rlhf_data = json.load(rlhf_file)

    dpo_data = []

    for dialogue in rlhf_data:
        conversation = {"conversations": []}
        chosen = {"from": "gpt", "value": ""}
        rejected = {"from": "gpt", "value": ""}

        user = 0
        assistant = 0
        for message in dialogue["messages"]:
            if message["role"] == "system":
                conversation["conversations"].append({"from": "system", "value": f"{message['content']}"})
            elif message["role"] == "user" and user == 0:
                conversation["conversations"].append({"from": "human", "value": f"{message['content']}"})
                user += 1
            elif message["role"] == "assistant" and assistant == 0:
                # Assign assistant responses as chosen; modify logic if you have rejected ones
                rejected["value"] = message["content"]
                assistant += 1
            elif message["role"] == "assistant" and assistant == 1:
                chosen["value"] = message["content"]

        # Add chosen and rejected fields to the conversation
        conversation["chosen"] = chosen
        conversation["rejected"] = rejected  # Include logic if rejection needs to be processed

        dpo_data.append(conversation)

    # Save the transformed data to a new file
    with open(dpo_file_path, 'w') as dpo_file:
        json.dump(dpo_data, dpo_file, indent=2)

# # Example usage:
# # Replace 'rlhf.json' with the path to your input file and 'dpo_output.json' with your desired output file path.
# convert_rlhf_to_dpo('../datasets/llama_pairwise.json', '../datasets/llama_dpo.json')
