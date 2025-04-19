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

# Loop through the list
for strategy in strategies:
    print(strategy)

    csv_path = "../datasets/human-impr/" + strategy + "_sft_improved_test_removed.csv"
    json_path = "../datasets/human-impr/" + strategy + "_sft_improved_test_removed.json"
    dataset_to_json_model_sft(csv_path, json_path)

    csv_path = "../datasets/human-impr/" + strategy + "_rlhf_improved_test_removed.csv"
    json_path = "../datasets/human-impr/" + strategy + "_rlhf_improved_pairwise_test_removed.json"
    dataset_to_json_model_rlhf(csv_path, json_path)

for strategy in strategies:
    print(strategy)

    json1_path = "../datasets/human-impr/" + strategy + "_rlhf_improved_pairwise_test_removed.json"
    json2_path = "../datasets/human-impr/" + strategy + "_rlhf_improved_dpo_test_removed.json"
    convert_rlhf_to_dpo(json1_path, json2_path)