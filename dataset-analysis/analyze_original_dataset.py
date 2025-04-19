import pandas as pd

# Load CSV file into a Pandas DataFrame
csv_file = "../datasets/dataset.csv"
df = pd.read_csv(csv_file)

print(f"rating_human == 1: {df[(df['rating_human'] == 1)].shape[0]}")
print(f"rating_human == 2: {df[(df['rating_human'] == 2)].shape[0]}")
print(f"rating_human == 3: {df[(df['rating_human'] == 3)].shape[0]}")

print(f"rating_chatgpt_empathy == 1: {df[(df['rating_chatgpt_empathy'] == 1)].shape[0]}")
print(f"rating_chatgpt_empathy == 2: {df[(df['rating_chatgpt_empathy'] == 2)].shape[0]}")
print(f"rating_chatgpt_empathy == 3: {df[(df['rating_chatgpt_empathy'] == 3)].shape[0]}")

print(f"rating_llama_empathy == 1: {df[(df['rating_llama_empathy'] == 1)].shape[0]}")
print(f"rating_llama_empathy == 2: {df[(df['rating_llama_empathy'] == 2)].shape[0]}")
print(f"rating_llama_empathy == 3: {df[(df['rating_llama_empathy'] == 3)].shape[0]}")

print(f"rating_gemini_empathy == 1: {df[(df['rating_gemini_empathy'] == 1)].shape[0]}")
print(f"rating_gemini_empathy == 2: {df[(df['rating_gemini_empathy'] == 2)].shape[0]}")
print(f"rating_gemini_empathy == 3: {df[(df['rating_gemini_empathy'] == 3)].shape[0]}")

print(f"rating_mixtral_empathy == 1: {df[(df['rating_mixtral_empathy'] == 1)].shape[0]}")
print(f"rating_mixtral_empathy == 2: {df[(df['rating_mixtral_empathy'] == 2)].shape[0]}")
print(f"rating_mixtral_empathy == 3: {df[(df['rating_mixtral_empathy'] == 3)].shape[0]}")

print("")

print(f"rating_human == 1 and rating_chatgpt_empathy == 1: {df[(df['rating_human'] == 1) & (df['rating_chatgpt_empathy'] == 1)].shape[0]}")
print(f"rating_human == 2 and rating_chatgpt_empathy == 1: {df[(df['rating_human'] == 2) & (df['rating_chatgpt_empathy'] == 1)].shape[0]}")
print(f"rating_human == 3 and rating_chatgpt_empathy == 1: {df[(df['rating_human'] == 3) & (df['rating_chatgpt_empathy'] == 1)].shape[0]}")
print(f"rating_human == 1 and rating_chatgpt_empathy == 2: {df[(df['rating_human'] == 1) & (df['rating_chatgpt_empathy'] == 2)].shape[0]}")
print(f"rating_human == 2 and rating_chatgpt_empathy == 2: {df[(df['rating_human'] == 2) & (df['rating_chatgpt_empathy'] == 2)].shape[0]}")
print(f"rating_human == 3 and rating_chatgpt_empathy == 2: {df[(df['rating_human'] == 3) & (df['rating_chatgpt_empathy'] == 2)].shape[0]}")
print(f"rating_human == 1 and rating_chatgpt_empathy == 3: {df[(df['rating_human'] == 1) & (df['rating_chatgpt_empathy'] == 3)].shape[0]}")
print(f"rating_human == 2 and rating_chatgpt_empathy == 3: {df[(df['rating_human'] == 2) & (df['rating_chatgpt_empathy'] == 3)].shape[0]}")
print(f"rating_human == 3 and rating_chatgpt_empathy == 3: {df[(df['rating_human'] == 3) & (df['rating_chatgpt_empathy'] == 3)].shape[0]}")

print(f"rating_human == 1 and rating_llama_empathy == 1: {df[(df['rating_human'] == 1) & (df['rating_llama_empathy'] == 1)].shape[0]}")
print(f"rating_human == 2 and rating_llama_empathy == 1: {df[(df['rating_human'] == 2) & (df['rating_llama_empathy'] == 1)].shape[0]}")
print(f"rating_human == 3 and rating_llama_empathy == 1: {df[(df['rating_human'] == 3) & (df['rating_llama_empathy'] == 1)].shape[0]}")
print(f"rating_human == 1 and rating_llama_empathy == 2: {df[(df['rating_human'] == 1) & (df['rating_llama_empathy'] == 2)].shape[0]}")
print(f"rating_human == 2 and rating_llama_empathy == 2: {df[(df['rating_human'] == 2) & (df['rating_llama_empathy'] == 2)].shape[0]}")
print(f"rating_human == 3 and rating_llama_empathy == 2: {df[(df['rating_human'] == 3) & (df['rating_llama_empathy'] == 2)].shape[0]}")
print(f"rating_human == 1 and rating_llama_empathy == 3: {df[(df['rating_human'] == 1) & (df['rating_llama_empathy'] == 3)].shape[0]}")
print(f"rating_human == 2 and rating_llama_empathy == 3: {df[(df['rating_human'] == 2) & (df['rating_llama_empathy'] == 3)].shape[0]}")
print(f"rating_human == 3 and rating_llama_empathy == 3: {df[(df['rating_human'] == 3) & (df['rating_llama_empathy'] == 3)].shape[0]}")

print(f"rating_human == 1 and rating_gemini_empathy == 1: {df[(df['rating_human'] == 1) & (df['rating_gemini_empathy'] == 1)].shape[0]}")
print(f"rating_human == 2 and rating_gemini_empathy == 1: {df[(df['rating_human'] == 2) & (df['rating_gemini_empathy'] == 1)].shape[0]}")
print(f"rating_human == 3 and rating_gemini_empathy == 1: {df[(df['rating_human'] == 3) & (df['rating_gemini_empathy'] == 1)].shape[0]}")
print(f"rating_human == 1 and rating_gemini_empathy == 2: {df[(df['rating_human'] == 1) & (df['rating_gemini_empathy'] == 2)].shape[0]}")
print(f"rating_human == 2 and rating_gemini_empathy == 2: {df[(df['rating_human'] == 2) & (df['rating_gemini_empathy'] == 2)].shape[0]}")
print(f"rating_human == 3 and rating_gemini_empathy == 2: {df[(df['rating_human'] == 3) & (df['rating_gemini_empathy'] == 2)].shape[0]}")
print(f"rating_human == 1 and rating_gemini_empathy == 3: {df[(df['rating_human'] == 1) & (df['rating_gemini_empathy'] == 3)].shape[0]}")
print(f"rating_human == 2 and rating_gemini_empathy == 3: {df[(df['rating_human'] == 2) & (df['rating_gemini_empathy'] == 3)].shape[0]}")
print(f"rating_human == 3 and rating_gemini_empathy == 3: {df[(df['rating_human'] == 3) & (df['rating_gemini_empathy'] == 3)].shape[0]}")

print(f"rating_human == 1 and rating_mixtral_empathy == 1: {df[(df['rating_human'] == 1) & (df['rating_mixtral_empathy'] == 1)].shape[0]}")
print(f"rating_human == 2 and rating_mixtral_empathy == 1: {df[(df['rating_human'] == 2) & (df['rating_mixtral_empathy'] == 1)].shape[0]}")
print(f"rating_human == 3 and rating_mixtral_empathy == 1: {df[(df['rating_human'] == 3) & (df['rating_mixtral_empathy'] == 1)].shape[0]}")
print(f"rating_human == 1 and rating_mixtral_empathy == 2: {df[(df['rating_human'] == 1) & (df['rating_mixtral_empathy'] == 2)].shape[0]}")
print(f"rating_human == 2 and rating_mixtral_empathy == 2: {df[(df['rating_human'] == 2) & (df['rating_mixtral_empathy'] == 2)].shape[0]}")
print(f"rating_human == 3 and rating_mixtral_empathy == 2: {df[(df['rating_human'] == 3) & (df['rating_mixtral_empathy'] == 2)].shape[0]}")
print(f"rating_human == 1 and rating_mixtral_empathy == 3: {df[(df['rating_human'] == 1) & (df['rating_mixtral_empathy'] == 3)].shape[0]}")
print(f"rating_human == 2 and rating_mixtral_empathy == 3: {df[(df['rating_human'] == 2) & (df['rating_mixtral_empathy'] == 3)].shape[0]}")
print(f"rating_human == 3 and rating_mixtral_empathy == 3: {df[(df['rating_human'] == 3) & (df['rating_mixtral_empathy'] == 3)].shape[0]}")

# print(f"rating_[llm]_empathy != 3: {df[(((df['rating_chatgpt_empathy'] != 3)) & ((df['rating_llama_empathy'] != 3)) & ((df['rating_gemini_empathy'] != 3)) & ((df['rating_mixtral_empathy'] != 3)))].shape[0]}")
# print(f"rating_human == [1,2] and rating_[llm]_empathy == [1, 2]: {df[((df['rating_human'] == 1) | (df['rating_human'] == 2)) & ((df['rating_chatgpt_empathy'] == 1) | (df['rating_chatgpt_empathy'] == 2)) & ((df['rating_llama_empathy'] == 1) | (df['rating_llama_empathy'] == 2)) & ((df['rating_gemini_empathy'] == 1) | (df['rating_gemini_empathy'] == 2)) & ((df['rating_mixtral_empathy'] == 1) | (df['rating_mixtral_empathy'] == 2))].shape[0]}")

print("")

matching_rows_count = 0 
# Loop through the rows using iterrows()
for index, row in df.iterrows():
#    if row[column_name] == condition_value:
    num_of_3 = 0
    if row["rating_human"] == 3:
           num_of_3 += 1
    if (row["rating_chatgpt_empathy"] == 3):
            num_of_3 += 1
    if (row["rating_llama_empathy"] == 3):
            num_of_3 += 1
    if (row["rating_gemini_empathy"] == 3):
            num_of_3 += 1
    if (row["rating_mixtral_empathy"] == 3):
            num_of_3 += 1
    if (num_of_3 <2): 
        matching_rows_count += 1  # Increment counter

print(f"Dialogues with fewer than 2 of score 3s: {matching_rows_count}")
