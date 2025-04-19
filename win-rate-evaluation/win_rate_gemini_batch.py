import pandas as pd  # Importing pandas for data manipulation
import importlib.util
from google import genai
import os  # Importing os module to access environment variables
import csv # Importing CSV to modify CSV files

def win_rate_gemini (model_1_config_path, model_2_config_path, win_rate_output_path): 
    # Dynamically import the configuration module
    spec = importlib.util.spec_from_file_location("model_1_config", model_1_config_path)
    model_1_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_1_config)

    spec = importlib.util.spec_from_file_location("model_2_config", model_2_config_path)
    model_2_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_2_config)

    # Set up the API key from an environment variable
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    # Reading the responses dataset
    test = pd.read_csv("../datasets/test.csv")
    model_1_response_file = pd.read_csv(model_1_config.OUTPUT_PATH_FILTERED)
    model_2_response_file = pd.read_csv(model_2_config.OUTPUT_PATH_FILTERED)
    context = test["situation"].to_list()  # Extracting context column as a list
    speaker_utterance = test["speaker_uttr"].to_list()  # Extracting speaker utterances
    response_1 = model_1_response_file["response"].to_list()  # Extracting response_1 column
    response_2 = model_2_response_file["response"].to_list()  # Extracting response_2 column

    # Defining a function to call GPT-4o and compare the empathy of two responses
    def gemini_call(context, speaker_utterance, response_1, response_2):

        prompt = f"""
        System Prompt: Respond only with 1 for Response 1 and 2 for Response 2. \
        Do not output any other content, just a 1 or a 2 for whichever response you believe to be the better empathetic response.

        User Prompt: Empathy is the ability to understand and share the feelings of another person. \
        It is the ability to put yourself in someone else’s shoes and see the world from their perspective. \
        Empathy is a complex skill that involves cognitive, emotional, and compassionate components. \
        Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. \
        It is being able to see the world through their eyes and understand their point of view. \
        Affective empathy is the ability to experience the emotions of another person. \
        It is feeling what they are feeling, both positive and negative. \
        Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed. \
        It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another’s distress or suffering. \
        Empathy is important because it allows us to connect with others on a deeper level. \
        It helps us to build trust, compassion, and intimacy. \
        Empathy is also essential for effective communication and conflict resolution. 

        You are given a dialogue context, a speaker utterance, and two responses to that speaker utterance in the dialogue context. 
        Compare the empathy of the following two responses and pick the better empathetic response. \
        Respond with 1 if Response 1 is the better empathetic response and 2 if Response 2 is the better empathetic response. 

        Context: {context} 

        Speaker Utterance: {speaker_utterance} 

        Response 1: {response_1} 

        Response 2: {response_2}
        """

        # print(prompt)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            # contents="Can you solve the following question: 1 + 2 = ?",
            contents=prompt
        )

        return str(response.text.strip())

    # List to store results
    results = []
    # Counters for model wins
    model_1_wins = 0
    model_2_wins = 0

    # Looping through all rows in the dataset to evaluate responses
    for i in range(len(test)):
        if ((i+1) % 50 == 0): 
            print(i + 1)

        # winner = gpt4o_call(context[i], speaker_utterance[i], response_1[i], response_2[i])  # Get the winning response
        winner = gemini_call(context[i], speaker_utterance[i], response_1[i], response_2[i])  # Get the winning response
        if winner == "1":
            # Append results for model 1 win
            results.append((context[i], speaker_utterance[i], response_1[i], response_2[i], "1", "0"))
            model_1_wins += 1
        elif winner == "2":
            # Append results for model 2 win
            results.append((context[i], speaker_utterance[i], response_1[i], response_2[i], "0", "1"))
            model_2_wins += 1
        else:
            # Handle irregular outputs
            print(f"Irregular output: {winner}")
            results.append((context[i], speaker_utterance[i], response_1[i], response_2[i], winner, winner))

    # ✅ Force cleanup to prevent HTTPX NoneType garbage collection error
    del client

    # Calculate win rates for both models
    model_1_win_rate = model_1_wins / len(test) * 100  # Model 1 win percentage
    model_2_win_rate = model_2_wins / len(test) * 100  # Model 2 win percentage

    # Print the win rates
    print(f"{model_1_config.MODEL} Win Rate: {model_1_win_rate:.2f}%")
    print(f"{model_2_config.MODEL} Win Rate: {model_2_win_rate:.2f}%")

    with open(win_rate_output_path, mode="w",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["context", "speaker_uttr", model_1_config.MODEL, model_2_config.MODEL, f"{model_1_config.MODEL}_win", f"{model_2_config.MODEL}_win"])
        for context, speaker_utterance, response_1, response_2, response_1_win, response_2_win in results:
            writer.writerow([context, speaker_utterance, response_1, response_2, response_1_win, response_2_win])
        writer.writerow(["", "", "", "", model_1_win_rate, model_2_win_rate])

# # Prompt the user to enter the path to the Python configuration file
# model_1_config_path = input("Enter the path to the Python configuration file for Model 1: ")
# model_2_config_path = input("Enter the path to the Python configuration file for Model 2: ")
# win_rate_output_path = input("Enter the path to the win rate output path: ")

# # Run win_rate_gemini 
# win_rate_gemini(model_1_config_path, model_2_config_path, win_rate_output_path)

base_model = "llama_3.1_8B"
base_model_path = "../config/llama_3.1_8B_base_config.py"

M1_model_paths = [
    # "../config/llama_3.1_8B_chatgpt_sft_config.py",
    # "../config/llama_3.1_8B_chatgpt_sft-dpo_config.py",
    # "../config/llama_3.1_8B_llama_sft_config.py",
    # "../config/llama_3.1_8B_llama_sft-dpo_config.py",
    # "../config/llama_3.1_8B_gemini_sft_config.py",
    # "../config/llama_3.1_8B_gemini_sft-dpo_config.py",
    # "../config/llama_3.1_8B_mixtral_sft_config.py",
    # "../config/llama_3.1_8B_mixtral_sft-dpo_config.py",
    # "../config/llama_3.1_8B_all_together_sft_config.py",
    # "../config/llama_3.1_8B_all_together_sft-dpo_config.py"
]

M2_model_paths = [
    # "../config/llama_3.1_8B_naive_prompt_sft_improved_config.py", 
    # "../config/llama_3.1_8B_naive_prompt_sft-dpo_improved_config.py", 
    # "../config/llama_3.1_8B_improve_all_three_dimensions_sft_improved_config.py", 
    # "../config/llama_3.1_8B_improve_all_three_dimensions_sft-dpo_improved_config.py",
    # "../config/llama_3.1_8B_improve_three_dimensions_sequentially_sft_improved_config.py",
    # "../config/llama_3.1_8B_improve_three_dimensions_sequentially_sft-dpo_improved_config.py",
    # "../config/llama_3.1_8B_identify_lacking_dimension_sft_improved_config.py", 
    # "../config/llama_3.1_8B_identify_lacking_dimension_sft-dpo_improved_config.py", 
    # "../config/llama_3.1_8B_improve_cognitive_sft_improved_config.py", 
    # "../config/llama_3.1_8B_improve_cognitive_sft-dpo_improved_config.py", 
    # "../config/llama_3.1_8B_improve_affective_sft_improved_config.py", 
    # "../config/llama_3.1_8B_improve_affective_sft-dpo_improved_config.py", 
    # "../config/llama_3.1_8B_improve_compassionate_sft_improved_config.py", 
    # "../config/llama_3.1_8B_improve_compassionate_sft-dpo_improved_config.py", 
    # "../config/llama_3.1_8B_combine_cognitive_affective_compassionate_sft_improved_config.py", 
    # "../config/llama_3.1_8B_combine_cognitive_affective_compassionate_sft-dpo_improved_config.py"
]

M3_model_paths = [
    # "../config/llama_3.1_8B_naive_prompt_improve_initial_response_sft_config.py", 
    # "../config/llama_3.1_8B_naive_prompt_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_improve_all_three_dimensions_improve_initial_response_sft_config.py",
    "../config/llama_3.1_8B_improve_all_three_dimensions_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_improve_three_dimensions_sequentially_improve_initial_response_sft_config.py",
    # "../config/llama_3.1_8B_improve_three_dimensions_sequentially_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_identify_lacking_dimension_improve_initial_response_sft_config.py",
    # "../config/llama_3.1_8B_identify_lacking_dimension_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_improve_cognitive_improve_initial_response_sft_config.py",
    # "../config/llama_3.1_8B_improve_cognitive_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_improve_affective_improve_initial_response_sft_config.py",
    # "../config/llama_3.1_8B_improve_affective_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_improve_compassionate_improve_initial_response_sft_config.py",
    # "../config/llama_3.1_8B_improve_compassionate_improve_initial_response_sft-dpo_config.py",
    # "../config/llama_3.1_8B_combined_improve_cognitive_improve_affective_improve_compassionate_improve_initial_response_sft_config.py",
    # "../config/llama_3.1_8B_combined_improve_cognitive_improve_affective_improve_compassionate_improve_initial_response_sft-dpo_config.py"
]

# base_model = "mistral_7B_v0.3"
# base_model_path = "../config/mistral_7B_v0.3_base_config.py"

# M1_model_paths = [
#     "../config/mistral_7B_v0.3_chatgpt_sft_config.py",
#     "../config/mistral_7B_v0.3_chatgpt_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_llama_sft_config.py",
#     "../config/mistral_7B_v0.3_llama_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_gemini_sft_config.py",
#     "../config/mistral_7B_v0.3_gemini_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_mixtral_sft_config.py",
#     "../config/mistral_7B_v0.3_mixtral_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_all_together_sft_config.py",
#     "../config/mistral_7B_v0.3_all_together_sft-dpo_config.py"
# ]

# M2_model_paths = [
#     "../config/mistral_7B_v0.3_naive_prompt_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_naive_prompt_sft-dpo_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_all_three_dimensions_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_all_three_dimensions_sft-dpo_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_three_dimensions_sequentially_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_three_dimensions_sequentially_sft-dpo_improved_config.py",
#     "../config/mistral_7B_v0.3_identify_lacking_dimension_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_identify_lacking_dimension_sft-dpo_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_cognitive_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_cognitive_sft-dpo_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_affective_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_affective_sft-dpo_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_compassionate_sft_improved_config.py",
#     "../config/mistral_7B_v0.3_improve_compassionate_sft-dpo_improved_config.py"
# ]

# M3_model_paths = [
#     "../config/mistral_7B_v0.3_naive_prompt_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_naive_prompt_improve_initial_response_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_improve_all_three_dimensions_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_improve_all_three_dimensions_improve_initial_response_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_improve_three_dimensions_sequentially_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_improve_three_dimensions_sequentially_improve_initial_response_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_identify_lacking_dimension_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_identify_lacking_dimension_improve_initial_response_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_improve_cognitive_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_improve_cognitive_improve_initial_response_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_improve_affective_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_improve_affective_improve_initial_response_sft-dpo_config.py",
#     "../config/mistral_7B_v0.3_improve_compassionate_improve_initial_response_sft_config.py",
#     "../config/mistral_7B_v0.3_improve_compassionate_improve_initial_response_sft-dpo_config.py"
# ]

for i in range(len(M1_model_paths)):
    model_1_config_path = base_model_path
    model_2_config_path = M1_model_paths[i]
    win_rate_output_path = "../datasets/gemini_win_rates/" \
                            + base_model \
                            + "_" \
                            + model_1_config_path.removeprefix("../config/llama_3.1_8B_").removesuffix("_config.py") \
                            + "_" \
                            + model_2_config_path.removeprefix("../config/llama_3.1_8B_").removesuffix("_config.py") \
                            + ".csv"
    # win_rate_output_path = "../datasets/gemini_win_rates/" \
    #                         + base_model \
    #                         + "_" \
    #                         + model_1_config_path.removeprefix("../config/mistral_7B_v0.3_").removesuffix("_config.py") \
    #                         + "_" \
    #                         + model_2_config_path.removeprefix("../config/mistral_7B_v0.3_").removesuffix("_config.py") \
    #                         + ".csv"

    print(model_1_config_path)
    print(model_2_config_path)
    print(win_rate_output_path)
    win_rate_gemini(model_1_config_path, model_2_config_path, win_rate_output_path)
    print('\n')

for i in range(len(M2_model_paths)):
    model_1_config_path = base_model_path
    model_2_config_path = M2_model_paths[i]
    win_rate_output_path = "../datasets/gemini_win_rates/" \
                            + base_model \
                            + "_" \
                            + model_1_config_path.removeprefix("../config/llama_3.1_8B_").removesuffix("_config.py") \
                            + "_" \
                            + model_2_config_path.removeprefix("../config/llama_3.1_8B_").removesuffix("_config.py") \
                            + ".csv"
    # win_rate_output_path = "../datasets/gemini_win_rates/" \
    #                         + base_model \
    #                         + "_" \
    #                         + model_1_config_path.removeprefix("../config/mistral_7B_v0.3_").removesuffix("_config.py") \
    #                         + "_" \
    #                         + model_2_config_path.removeprefix("../config/mistral_7B_v0.3_").removesuffix("_config.py") \
    #                         + ".csv"

    print(model_1_config_path)
    print(model_2_config_path)
    print(win_rate_output_path)
    win_rate_gemini(model_1_config_path, model_2_config_path, win_rate_output_path)
    print('\n')

for i in range(len(M3_model_paths)):
    model_1_config_path = base_model_path
    model_2_config_path = M3_model_paths[i]
    win_rate_output_path = "../datasets/gemini_win_rates/" \
                            + base_model \
                            + "_" \
                            + model_1_config_path.removeprefix("../config/llama_3.1_8B_").removesuffix("_config.py") \
                            + "_" \
                            + model_2_config_path.removeprefix("../config/llama_3.1_8B_").removesuffix("_config.py") \
                            + ".csv"
    # win_rate_output_path = "../datasets/gemini_win_rates/" \
    #                         + base_model \
    #                         + "_" \
    #                         + model_1_config_path.removeprefix("../config/mistral_7B_v0.3_").removesuffix("_config.py") \
    #                         + "_" \
    #                         + model_2_config_path.removeprefix("../config/mistral_7B_v0.3_").removesuffix("_config.py") \
    #                         + ".csv"

    print(model_1_config_path)
    print(model_2_config_path)
    print(win_rate_output_path)
    win_rate_gemini(model_1_config_path, model_2_config_path, win_rate_output_path)
    print('\n')