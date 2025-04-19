from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas
import importlib.util
import csv

# Prompt the user to enter the path to the Python configuration file
config_path = input("Enter the path to the Python configuration file for the base model: ")

# Dynamically import the configuration module
spec = importlib.util.spec_from_file_location("base_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Define paths for base model and PEFT models
base_path = config.MODEL_PATH

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

test_file = pandas.read_csv("../datasets/test.csv")
context = test_file["situation"].to_list()
speaker_utterance = test_file["speaker_uttr"].to_list()
# Loop over the test data to generate responses
with open(config.OUTPUT_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["context", "speaker_uttr", "response"])
    for i in range(len(test_file)):
        print(i + 1)

        # Define a prompt for the conversation, combining instructions and dynamic inputs
        # prompt = "Hey, are you conscious? Can you talk to me?"
        prompt = "Empathy is the ability to understand and share the feelings of another person. " + \
                        "It is the ability to put yourself in someone else's shoes and see the world from their perspective. " + \
                        "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. " + \
                        "Cognitive empathy is the ability to understand another person's thoughts, beliefs, and intentions. " + \
                        "It is being able to see the world through their eyes and understand their point of view. " + \
                        "Affective empathy is the ability to experience the emotions of another person. " + \
                        "It is feeling what they are feeling, both positive and negative. " + \
                        "Compassionate empathy is the ability to not only understand and share another person's feelings, but also to be moved to help if needed. " + \
                        "It involves a deeper level of emotional engagement than cognitive empathy, prompting action to alleviate another's distress or suffering. " + \
                        "Empathy is important because it allows us to connect with others on a deeper level. " + \
                        "It helps us to build trust, compassion, and intimacy. " + \
                        "Empathy is also essential for effective communication and conflict resolution. " + \
                        "You are engaging in a conversation with a human. " + \
                        "Respond in an empathetic manner to the following speaker utterance in a given context using on average 28 words and a maximum of 97 words.\n" + \
                        f"Context: {context[i]} \n" + \
                        f"Speaker Utterance: {speaker_utterance[i]} \n" + \
                        "Response: "

        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        generated_ids = model.generate(**model_inputs, max_new_tokens=130, do_sample=True)
        response = tokenizer.batch_decode(generated_ids)[0]
        
        print(response)
        print("")
        print("---------------------")
        print("")

        writer.writerow([context[i], speaker_utterance[i], response])