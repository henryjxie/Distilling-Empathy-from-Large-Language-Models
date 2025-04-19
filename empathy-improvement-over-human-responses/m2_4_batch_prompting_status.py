from openai import OpenAI
import os
import json

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# SFT
with open("datasets/naive_prompt_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    
    with open("datasets/naive_prompt_improve_initial_response_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/improve_all_three_dimensions_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_all_three_dimensions_improve_initial_response_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/identify_lacking_dimension_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/identify_lacking_dimension_improve_initial_response_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/improve_cognitive_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_cognitive_improve_initial_response_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/improve_affective_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_affective_improve_initial_response_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)
    
with open("datasets/improve_compassionate_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_compassionate_improve_initial_response_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

# RLHF
with open("datasets/naive_prompt_rlhf_batch_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    
    with open("datasets/naive_prompt_rlhf_batch_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/improve_all_three_dimensions_rlhf_batch_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_all_three_dimensions_rlhf_batch_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/identify_lacking_dimension_rlhf_batch_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/identify_lacking_dimension_rlhf_batch_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/improve_cognitive_rlhf_batch_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_cognitive_rlhf_batch_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)

with open("datasets/improve_affective_rlhf_batch_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_affective_rlhf_batch_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)
    
with open("datasets/improve_compassionate_rlhf_batch_object.json") as file:
    data = json.load(file)
    print(client.batches.retrieve(data["id"]).to_dict()["status"])
    client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"])

    with open("datasets/improve_compassionate_rlhf_batch_results.jsonl", "wb") as file:
        file.write(client.files.content(client.batches.retrieve(data["id"]).to_dict()["output_file_id"]).content)