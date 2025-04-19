import pandas

def remove_test(test_file, input_file, output_file): 

    test = pandas.read_csv(test_file)
    test_dialog_id = test["dialog_id"].to_list()

    dataset = pandas.read_csv(input_file)
    print(len(dataset))
    dataset_dialog_id = dataset["dialog_id"].to_list()

    for i in range(len(dataset_dialog_id)):
        if dataset_dialog_id[i] in test_dialog_id:
            dataset = dataset.drop(i)

    print(len(dataset))
    dataset.to_csv(output_file, index=False)


remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/naive_prompt_improve_initial_response.csv", 
            "../datasets/llm-impr/naive_prompt_improve_initial_response.csv")


remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/improve_all_three_dimensions_improve_initial_response.csv", 
            "../datasets/llm-impr/improve_all_three_dimensions_improve_initial_response.csv") 

remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/identify_lacking_dimension_improve_initial_response.csv", 
            "../datasets/llm-impr/identify_lacking_dimension_improve_initial_response.csv")

remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/improve_cognitive_improve_initial_response.csv", 
            "../datasets/llm-impr/improve_cognitive_improve_initial_response.csv")

remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/improve_affective_improve_initial_response.csv", 
            "../datasets/llm-impr/improve_affective_improve_initial_response.csv")

remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/improve_compassionate_improve_initial_response.csv", 
            "../datasets/llm-impr/improve_compassionate_improve_initial_response.csv")

remove_test("../datasets/test.csv", 
            "../datasets/llm-impr/improve_three_dimensions_sequentially_improve_initial_response.csv", 
            "../datasets/llm-impr/improve_three_dimensions_sequentially_improve_initial_response.csv")
