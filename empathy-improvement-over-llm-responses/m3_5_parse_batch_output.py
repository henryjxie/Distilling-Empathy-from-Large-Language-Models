import json
import pandas

# SFT
original_dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

data = []
with open("../datasets/llm-impr/naive_prompt_improve_initial_response_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_initial_response"] = data
    original_dataset.to_csv("../datasets/llm-impr/naive_prompt_improve_initial_response.csv", index=False)

original_dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

data = []
with open("../datasets/llm-impr/improve_all_three_dimensions_improve_initial_response_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_initial_response"] = data
    original_dataset.to_csv("../datasets/llm-impr/improve_all_three_dimensions_improve_initial_response.csv", index=False)

original_dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

data = []
with open("../datasets/llm-impr/identify_lacking_dimension_improve_initial_response_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_initial_response"] = data
    original_dataset.to_csv("../datasets/llm-impr/identify_lacking_dimension_improve_initial_response.csv", index=False)

original_dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

data = []
with open("../datasets/llm-impr/improve_cognitive_improve_initial_response_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_initial_response"] = data
    original_dataset.to_csv("../datasets/llm-impr/improve_cognitive_improve_initial_response.csv", index=False)

original_dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

data = []
with open("../datasets/llm-impr/improve_affective_improve_initial_response_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_initial_response"] = data
    original_dataset.to_csv("../datasets/llm-impr/improve_affective_improve_initial_response.csv", index=False)

original_dataset = pandas.read_csv("../datasets/llm-impr/generate_initial_response.csv")

data = []
with open("../datasets/llm-impr/improve_compassionate_improve_initial_response_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_initial_response"] = data
    original_dataset.to_csv("../datasets/llm-impr/improve_compassionate_improve_initial_response.csv", index=False)

