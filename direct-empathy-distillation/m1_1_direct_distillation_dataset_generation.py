import pandas
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.abspath('./utils'))

from dataset_to_json_model_sft import dataset_to_json_model_sft
from dataset_to_json_model_rlhf import dataset_to_json_model_rlhf
from pairwise_to_dpo import convert_rlhf_to_dpo

dataset = pandas.read_csv("../datasets/train.csv")

# Creating the sft dataset for each llm
chatgpt_sft_rows = dataset[((dataset["rating_human"]) == 3) & (dataset["rating_chatgpt_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_chatgpt_empathy"]]
chatgpt_sft_rows.rename(columns={"response_chatgpt_empathy": "response_model"}, inplace=True)

llama_sft_rows = dataset[((dataset["rating_human"]) == 3) & (dataset["rating_llama_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_llama_empathy"]]
llama_sft_rows.rename(columns={"response_llama_empathy": "response_model"}, inplace=True)

gemini_sft_rows = dataset[((dataset["rating_human"]) == 3) & (dataset["rating_gemini_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_gemini_empathy"]]
gemini_sft_rows.rename(columns={"response_gemini_empathy": "response_model"}, inplace=True)

mixtral_sft_rows = dataset[((dataset["rating_human"]) == 3) & (dataset["rating_mixtral_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_mixtral_empathy"]]
mixtral_sft_rows.rename(columns={"response_mixtral_empathy": "response_model"}, inplace=True)

chatgpt_sft_rows.to_csv("../datasets/direct-dist/chatgpt_sft.csv", index=False)
llama_sft_rows.to_csv("../datasets/direct-dist/llama_sft.csv", index=False)
gemini_sft_rows.to_csv("../datasets/direct-dist/gemini_sft.csv", index=False)
mixtral_sft_rows.to_csv("../datasets/direct-dist/mixtral_sft.csv", index=False)

# Creating the rlhf dataset for each llm
chatgpt_rlhf_rows = dataset[((dataset["rating_human"]) == 2) | (dataset["rating_human"] == 1) & (dataset["rating_chatgpt_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_chatgpt_empathy"]]
chatgpt_rlhf_rows.rename(columns={"response_chatgpt_empathy": "response_model"}, inplace=True)

llama_rlhf_rows = dataset[((dataset["rating_human"]) == 2) | (dataset["rating_human"] == 1) & (dataset["rating_llama_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_llama_empathy"]]
llama_rlhf_rows.rename(columns={"response_llama_empathy": "response_model"}, inplace=True)

gemini_rlhf_rows = dataset[((dataset["rating_human"] == 2) | (dataset["rating_human"] == 1) & (dataset["rating_gemini_empathy"] == 3))][["dialog_id", "situation", "speaker_uttr", "response_human", "response_gemini_empathy"]]
gemini_rlhf_rows.rename(columns={"response_gemini_empathy": "response_model"}, inplace=True)

mixtral_rlhf_rows = dataset[((dataset["rating_human"]) == 2) | (dataset["rating_human"] == 1) & (dataset["rating_mixtral_empathy"] == 3)][["dialog_id", "situation", "speaker_uttr", "response_human", "response_mixtral_empathy"]]
mixtral_rlhf_rows.rename(columns={"response_mixtral_empathy": "response_model"}, inplace=True)

chatgpt_rlhf_rows.to_csv("../datasets/direct-dist/chatgpt_rlhf.csv", index=False)
llama_rlhf_rows.to_csv("../datasets/direct-dist/llama_rlhf.csv", index=False)
gemini_rlhf_rows.to_csv("../datasets/direct-dist/gemini_rlhf.csv", index=False)
mixtral_rlhf_rows.to_csv("../datasets/direct-dist/mixtral_rlhf.csv", index=False)

# Combine sft datasets
chatgpt_sft = pandas.read_csv("../datasets/direct-dist/chatgpt_sft.csv")
llama_sft = pandas.read_csv("../datasets/direct-dist/llama_sft.csv")
gemini_sft = pandas.read_csv("../datasets/direct-dist/gemini_sft.csv")
mixtral_sft = pandas.read_csv("../datasets/direct-dist/mixtral_sft.csv")

combined_sft = pandas.concat([chatgpt_sft, llama_sft, gemini_sft, mixtral_sft])
combined_sft.to_csv("../datasets/direct-dist/all_together_sft.csv", index=False)

# Combin rlhf datasets
chatgpt_rlhf = pandas.read_csv("../datasets/direct-dist/chatgpt_rlhf.csv")
llama_rlhf = pandas.read_csv("../datasets/direct-dist/llama_rlhf.csv")
gemini_rlhf = pandas.read_csv("../datasets/direct-dist/gemini_rlhf.csv")
mixtral_rlhf = pandas.read_csv("../datasets/direct-dist/mixtral_rlhf.csv")

combined_rlhf = pandas.concat([chatgpt_rlhf, llama_rlhf, gemini_rlhf, mixtral_rlhf])
combined_rlhf.to_csv("../datasets/direct-dist/all_together_rlhf.csv", index=False)

# Convert to correct format for llamafactory to consume 

# Create a list of models
models = ['chatgpt', 'llama', 'gemini', 'mixtral', 'all_together']

# Loop through the list
for model in models:
    print(model)

    csv_path = "../datasets/direct-dist/" + model + "_sft.csv"
    json_path = "../datasets/direct-dist/" + model + "_sft.json"
    dataset_to_json_model_sft(csv_path, json_path)

    csv_path = "../datasets/direct-dist/" + model + "_rlhf.csv"
    json_path = "../datasets/direct-dist/" + model + "_pairwise.json"
    dataset_to_json_model_rlhf(csv_path, json_path)

for model in models:
    print(model)

    json1_path = "../datasets/direct-dist/" + model + "_pairwise.json"
    json2_path = "../datasets/direct-dist/" + model + "_dpo.json"
    convert_rlhf_to_dpo(json1_path, json2_path)