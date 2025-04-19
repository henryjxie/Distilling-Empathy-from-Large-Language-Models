import pandas as pd  # Importing pandas for data manipulation
import importlib.util
from openai import OpenAI  # Importing OpenAI API for GPT-4o calls
import os  # Importing os module to access environment variables
import csv # Importing CSV to modify CSV files

# Prompt the user to enter the path to the Python configuration file
model_1_config_path = input("Enter the path to the Python configuration file for Model 1: ")

# Dynamically import the configuration module
spec = importlib.util.spec_from_file_location("model_1_config", model_1_config_path)
model_1_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_1_config)

model_2_config_path = input("Enter the path to the Python configuration file for Model 2: ")

spec = importlib.util.spec_from_file_location("model_2_config", model_2_config_path)
model_2_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_2_config)

win_rate_output_path = input("Enter the path to the win rate output path: ")

# Retrieving OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initializing the OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)

# Reading the responses dataset
test = pd.read_csv("../datasets/test.csv")
model_1_response_file = pd.read_csv(model_1_config.OUTPUT_PATH_FILTERED)
model_2_response_file = pd.read_csv(model_2_config.OUTPUT_PATH_FILTERED)
context = test["situation"].to_list()  # Extracting context column as a list
speaker_utterance = test["speaker_uttr"].to_list()  # Extracting speaker utterances
response_1 = model_1_response_file["response"].to_list()  # Extracting response_1 column
response_2 = model_2_response_file["response"].to_list()  # Extracting response_2 column

# Defining a function to call GPT-4o and compare the empathy of two responses
def gpt4o_call(context, speaker_utterance, response_1, response_2):
    prompt = [
        {"role": "system", "content": "Respond only with 1 for Response 1 and 2 for Response 2. " + \
        "Do not output any other content, just a 1 or a 2 for whichever response you believe to be the better empathetic response."},
        {"role": "user", "content": "Empathy is the ability to understand and share the feelings of another person. " + \
            "It is the ability to put yourself in someone else’s shoes and see the world from their perspective. " + \
            "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. " + \
            "Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. " + \
            "It is being able to see the world through their eyes and understand their point of view. " + \
            "Affective empathy is the ability to experience the emotions of another person. " + \
            "It is feeling what they are feeling, both positive and negative. " + \
            "Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed. " + \
            "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another’s distress or suffering. " + \
            "Empathy is important because it allows us to connect with others on a deeper level. " + \
            "It helps us to build trust, compassion, and intimacy. " + \
            "Empathy is also essential for effective communication and conflict resolution. " + \
            "You are given a dialogue context, a speaker utterance, and two responses to that speaker utterance in the dialogue context. "
            "Compare the empathy of the following two responses and pick the better empathetic response. " + \
            "Respond with 1 if Response 1 is the better empathetic response and 2 if Response 2 is the better empathetic response. \n" + \
            f"Context: {context} \n" + \
            f"Speaker Utterance: {speaker_utterance} \n"
            f"Response 1: {response_1} \n" + \
            f"Response 2: {response_2}"}
        ]

    # Making a call to the GPT-4o model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt
    )

    # Extracting the response choice (1 or 2)
    response = str(response.choices[0].message.content)

    return response

# List to store results
results = []
# Counters for model wins
model_1_wins = 0
model_2_wins = 0

# Looping through all rows in the dataset to evaluate responses
for i in range(len(test)):
    print(i + 1)

    winner = gpt4o_call(context[i], speaker_utterance[i], response_1[i], response_2[i])  # Get the winning response
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