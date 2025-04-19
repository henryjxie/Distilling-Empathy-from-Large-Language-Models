import importlib.util
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import pandas as pd
import csv

# Prompt the user to enter the path to the Python configuration file
config_path = input("Enter the path to the Python configuration file for the base model: ")

# Dynamically import the configuration module
spec = importlib.util.spec_from_file_location("base_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Define paths for base model and PEFT models
base_path = config.MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(base_path)

# Load the base model with proper settings for dtype and device mapping
model = LlamaForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set models to evaluation mode to disable training-specific layers like dropout
model.eval()

# Load the test data and extract context and speaker utterance columns
test_file = pd.read_csv("../datasets/test.csv")
context = test_file["situation"].to_list()
speaker_utterance = test_file["speaker_uttr"].to_list()

# Loop over the test data to generate responses (currently set to process one sample)
with open(config.OUTPUT_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["context", "speaker_uttr", "response"])
    for i in range(len(test_file)):
        print(i + 1)

        # Define a prompt for the conversation, combining instructions and dynamic inputs
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
                "Respond in an empathetic manner to the following speaker utterance in a given context using on average 28 words and a maximum of 97 words." + \
                "Only generate one succint response once. " + \
                "Do not output any extra information, just the response to the prompt.\n" + \
                f"Context: {context[i]} \n" + \
                f"Speaker Utterance: {speaker_utterance[i]} \n" + \
                "Response: "
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Tokenize the input prompt for both models
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        with torch.no_grad():
            output_ids_1 = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=4096
            )

        response = tokenizer.decode(output_ids_1[0], skip_special_tokens=True)

        print(response)
        print("")
        print("---------------------")
        print("")

        writer.writerow([context[i], speaker_utterance[i], response])
