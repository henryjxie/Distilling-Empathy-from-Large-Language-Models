import pandas as pd
import csv

# Load the test data and extract context and speaker utterance columns
test_file = pd.read_csv("../datasets/test.csv")
context = test_file["situation"].to_list()
speaker_utterance = test_file["speaker_uttr"].to_list()

OUTPUT_PATH = "../datasets/gpt4o/gpt4o-test-responses.csv"

# Load the initial responses from gpt-4o
gpt4o_responses_file = pd.read_csv("../datasets/llm-impr/generate_initial_response.csv") 

# Loop over the test data to generate responses (currently set to process one sample)
with open(OUTPUT_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["context", "speaker_uttr", "response"])
    for i in range(len(test_file)):
        print(i + 1)

        # response = gpt4o_responses_file[(gpt4o_responses_file["context"] == context[i]) & (gpt4o_responses_file["speaker_uttr"] == speaker_utterance[i])]["initial_response"]
        filtered = gpt4o_responses_file[
            (gpt4o_responses_file["situation"] == context[i]) &
            (gpt4o_responses_file["speaker_uttr"] == speaker_utterance[i])
        ]

        if not filtered.empty:
            response = filtered["initial_response"].values[0]
        else:
            response = None  # or handle however you like

        print(response)
        print("")
        print("---------------------")
        print("")

        writer.writerow([context[i], speaker_utterance[i], response])