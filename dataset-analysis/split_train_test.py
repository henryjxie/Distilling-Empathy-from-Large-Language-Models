import pandas as pd

# Load CSV file into a Pandas DataFrame
csv_file = "../datasets/dataset.csv"
df = pd.read_csv(csv_file)

train_df = pd.DataFrame(columns=df.columns)
test_df = pd.DataFrame(columns=df.columns)

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
        test_df.loc[len(test_df)] = row # Add the row to the test df

combined = pd.concat([df, test_df])
train_df = combined.drop_duplicates(keep=False)

train_df.to_csv("../datasets/train.csv", index=False)
test_df.to_csv('../datasets/test.csv', index=False)

