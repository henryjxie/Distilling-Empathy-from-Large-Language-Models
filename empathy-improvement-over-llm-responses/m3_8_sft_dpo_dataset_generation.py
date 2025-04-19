import pandas
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.abspath('./utils'))

# Add the utils/ directory to Python's module search path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

from dataset_to_json_model_sft import dataset_to_json_model_sft
from dataset_to_json_model_rlhf import dataset_to_json_model_rlhf
from pairwise_to_dpo import convert_rlhf_to_dpo

# Create a list of strategies
strategies = ['naive_prompt', 
          'improve_all_three_dimensions', 
          'identify_lacking_dimension', 
          'improve_cognitive',
          'improve_affective',
          'improve_compassionate',
          'improve_three_dimensions_sequentially']

for strategy in strategies:
    print(strategy)

    dataset = pandas.read_csv("../datasets/llm-impr/" + strategy + "_improve_initial_response.csv")

    sft_dataset = dataset.sample(n=950, random_state=42)
    rlhf_dataset = dataset.drop(sft_dataset.index)

    sft_dataset.to_csv("../datasets/llm-impr/" + strategy + "_improve_initial_response_sft.csv", index=False)
    rlhf_dataset.to_csv("../datasets/llm-impr/" + strategy + "_improve_initial_response_rlhf.csv", index=False)

# Loop through the list
for strategy in strategies:
    print(strategy)

    csv_path = "../datasets/llm-impr/" + strategy + "_improve_initial_response_sft.csv"
    json_path = "../datasets/llm-impr/" + strategy + "_improve_initial_response_sft.json"
    dataset_to_json_model_sft(csv_path, json_path)

    csv_path = "../datasets/llm-impr/" + strategy + "_improve_initial_response_rlhf.csv"
    json_path = "../datasets/llm-impr/" + strategy + "_improve_initial_response_pairwise.json"
    dataset_to_json_model_rlhf(csv_path, json_path)

for strategy in strategies:
    print(strategy)

    json1_path = "../datasets/llm-impr/" + strategy + "_improve_initial_response_pairwise.json"
    json2_path = "../datasets/llm-impr/" + strategy + "_improve_initial_response_dpo.json"
    convert_rlhf_to_dpo(json1_path, json2_path)