import pandas

def remove_test(test_file, input_file, output_file): 

    test = pandas.read_csv(test_file)
    test_dialog_id = test["dialog_id"].to_list()

    dataset = pandas.read_csv(input_file)
    # print(len(dataset))
    dataset_dialog_id = dataset["dialog_id"].to_list()

    for i in range(len(dataset_dialog_id)):
        if dataset_dialog_id[i] in test_dialog_id:
            dataset = dataset.drop(i)

    # print(len(dataset))
    dataset.to_csv(output_file, index=False)

#SFT
remove_test("../datasets/test.csv", 
            "../datasets/human-impr/naive_prompt_sft_improved.csv", 
            "../datasets/human-impr/naive_prompt_sft_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_all_three_dimensions_sft_improved.csv", 
            "../datasets/human-impr/improve_all_three_dimensions_sft_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/identify_lacking_dimension_sft_improved.csv", 
            "../datasets/human-impr/identify_lacking_dimension_sft_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_cognitive_sft_improved.csv", 
            "../datasets/human-impr/improve_cognitive_sft_improved_test_removed.csv")
     
remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_affective_sft_improved.csv", 
            "../datasets/human-impr/improve_affective_sft_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_compassionate_sft_improved.csv", 
            "../datasets/human-impr/improve_compassionate_sft_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_three_dimensions_sequentially_sft_improved.csv", 
            "../datasets/human-impr/improve_three_dimensions_sequentially_sft_improved_test_removed.csv")

#RLHF
remove_test("../datasets/test.csv", 
            "../datasets/human-impr/naive_prompt_rlhf_improved.csv", 
            "../datasets/human-impr/naive_prompt_rlhf_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_all_three_dimensions_rlhf_improved.csv", 
            "../datasets/human-impr/improve_all_three_dimensions_rlhf_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/identify_lacking_dimension_rlhf_improved.csv", 
            "../datasets/human-impr/identify_lacking_dimension_rlhf_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_cognitive_rlhf_improved.csv", 
            "../datasets/human-impr/improve_cognitive_rlhf_improved_test_removed.csv")
     
remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_affective_rlhf_improved.csv", 
            "../datasets/human-impr/improve_affective_rlhf_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_compassionate_rlhf_improved.csv", 
            "../datasets/human-impr/improve_compassionate_rlhf_improved_test_removed.csv")

remove_test("../datasets/test.csv", 
            "../datasets/human-impr/improve_three_dimensions_sequentially_rlhf_improved.csv", 
            "../datasets/human-impr/improve_three_dimensions_sequentially_rlhf_improved_test_removed.csv")