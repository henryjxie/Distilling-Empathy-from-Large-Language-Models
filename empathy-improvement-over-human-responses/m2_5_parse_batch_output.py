import json
import pandas

# SFT
original_dataset = pandas.read_csv("../datasets/human-impr/human_sft_dataset.csv")

data = []
with open("../datasets/human-impr/naive_prompt_sft_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/naive_prompt_sft_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_sft_dataset.csv")

data = []
with open("../datasets/human-impr/improve_all_three_dimensions_sft_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_all_three_dimensions_sft_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_sft_dataset.csv")

data = []
with open("../datasets/human-impr/identify_lacking_dimension_sft_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/identify_lacking_dimension_sft_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_sft_dataset.csv")

data = []
with open("../datasets/human-impr/improve_cognitive_sft_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_cognitive_sft_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_sft_dataset.csv")

data = []
with open("../datasets/human-impr/improve_affective_sft_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_affective_sft_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_sft_dataset.csv")

data = []
with open("../datasets/human-impr/improve_compassionate_sft_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_compassionate_sft_improved.csv", index=False)


# RLHF
original_dataset = pandas.read_csv("../datasets/human-impr/human_rlhf_dataset.csv")

data = []
with open("../datasets/human-impr/naive_prompt_rlhf_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/naive_prompt_rlhf_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_rlhf_dataset.csv")

data = []
with open("../datasets/human-impr/improve_all_three_dimensions_rlhf_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_all_three_dimensions_rlhf_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_rlhf_dataset.csv")

data = []
with open("../datasets/human-impr/identify_lacking_dimension_rlhf_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/identify_lacking_dimension_rlhf_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_rlhf_dataset.csv")

data = []
with open("../datasets/human-impr/improve_cognitive_rlhf_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_cognitive_rlhf_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_rlhf_dataset.csv")

data = []
with open("../datasets/human-impr/improve_affective_rlhf_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_affective_rlhf_improved.csv", index=False)

original_dataset = pandas.read_csv("../datasets/human-impr/human_rlhf_dataset.csv")

data = []
with open("../datasets/human-impr/improve_compassionate_rlhf_batch_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        response = json.loads(line)
        response = response["response"]["body"]["choices"][0]["message"]["content"]
        data.append(response)
    original_dataset["improved_human_response"] = data
    original_dataset.to_csv("../datasets/human-impr/improve_compassionate_rlhf_improved.csv", index=False)
