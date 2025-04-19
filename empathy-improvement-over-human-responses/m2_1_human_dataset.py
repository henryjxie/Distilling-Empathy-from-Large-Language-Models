import pandas

dataset = pandas.read_csv("../datasets/dataset.csv")

sft_dataset = pandas.DataFrame(columns=dataset.columns)
rlhf_dataset = pandas.DataFrame(columns=dataset.columns)

for i, row in dataset.iterrows():
    if row["rating_human"] == 3:
        sft_dataset.loc[len(sft_dataset)] = row
    elif row["rating_human"] == 1 or row["rating_human"] == 2:
        rlhf_dataset.loc[len(rlhf_dataset)] = row

sft_dataset.to_csv("../datasets/human-impr/human_sft_dataset.csv")
rlhf_dataset.to_csv("../datasets/human-impr/human_rlhf_dataset.csv")