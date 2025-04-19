# Distilling Empathy from Large Language Models
Henry Xie

April 21, 2025

## Overview
The distillation of knowledge from Large Language Models (LLMs) into Small Language Models (SLMs), preserving the capabilities and performance of LLMs while reducing model sizes, has played a key role in the proliferation of LLMs. 
Because SLMs are considerably smaller than LLMs, they are often utilized in domains where human interaction is necessary while resources are highly constrained, e.g., smart phones. 
Therefore, it is crucial to ensure that empathy, a fundamental aspect of positive human interactions, already instilled into LLMs, is retained by SLMs after distillation.
In this paper, we develop a comprehensive approach for effective empathy distillation from LLMs into SLMs. Our approach features a two-step fine-tuning process, fully leveraging datasets of empathetic dialogue responses distilled from LLMs. 
We explore several distillation methods beyond basic direct prompting and propose four unique sets of prompts for targeted empathy improvement to significantly enhance the empathy distillation process. 
Our evaluations demonstrate that SLMs fine-tuned through the two-step fine-tuning process, with distillation datasets enhanced by the targeted empathy improvement prompts, significantly outperform the base SLM at generating empathetic responses, with a win rate of 90+\%. 
Our targeted empathy improvement prompts substantially outperform the basic direct prompting with a 10\% improvement in win rate.

## About this Repository
This repository contains the code for our study on methods for effective empathy distillation.

####  Folders
`config`: This folder contains the example configuration files for base LLaMA-3.1-8B, Mistral-7B-v0.3, and GPT-4o.

`dataset-analysis: This folder provides two Python files, one for analyzing the statistics of the dataset and the other for creating the train and test dataset from the original dataset.

`datasets`: This folder contains files, one being the original dataset, and the other for creating the directories of the project.

`direct-empathy-distillation`: This folder contains a util folder and a Python file. The Python file is for creating the datasets for direct empathy distillation, compiling the responses for each model dataset. The util folder contains Python programs for converting CSV datasets to LLaMA-Factory dataset format.

`empathy-improvement-over-human-responses`: This folder contains a util folder and Python programs numbered as a step-by-step process. The util folder contains Python programs for converting CSV datasets to LLaMA-Factory dataset format.
* m2_1_human_dataset.py - This Python program creates the human SFT and RLHF datasets, splitting the human responses based on their empathy score.
* m2_2_create_batch_prompting_datasets.py - This Python program creates the datasets, in OpenAI API batch prompting format, for each of the Targeted Empathy Improvement prompting strategies.
* m2_3_batch_prompting - This Python program utilizes the OpenAI API batch prompting to generate the responses for each Targeted Empathy Improvement prompting strategy.
* m2_4_batch_prompting_status - This Python program utilizes the OpenAI API batch prompting to receive the status of each of the batch requests, generating batch objects for each request and downloading the results.
* m2_5_parse_batch_output.py - This Python program parses the output of the OpenAI API batch prompting process.
* m2_6_improve_three_dimensions_sequentially.py - This Python program runs the process for the Targeted Empathy Improvement prompting strategy of improving each of the three dimensions sequentially.
* m2_7_remove_test.py - This Python program removes the test dialogues from the train datasets.
* m2_8_sft_dpo_dataset_generation.py - This Python program converts the CSV datasets to LLaMA-Factory dataset format.

`empathy-improvement-over-llm-responses`: This folder contains a util folder and Python programs numbered as a step-by-step process. The util folder contains Python programs for converting CSV datasets to LLaMA-Factory dataset format.
* m2_1_batch_generate_initial_response.py - This Python program employs OpenAI API batch prompting to generate initial responses to each of the train dialogues.
* m2_2_create_batch_prompting_datasets.py - This Python program creates the datasets, in OpenAI API batch prompting format, for each of the Targeted Empathy Improvement prompting strategies.
* m2_3_batch_prompting - This Python program utilizes the OpenAI API batch prompting to generate the responses for each Targeted Empathy Improvement prompting strategy.
* m2_4_batch_prompting_status - This Python program utilizes the OpenAI API batch prompting to receive the status of each of the batch requests, generating batch objects for each request and downloading the results.
* m2_5_parse_batch_output.py - This Python program parses the output of the OpenAI API batch prompting process.
* m2_6_improve_three_dimensions_sequentially.py - This Python program runs the process for the Targeted Empathy Improvement prompting strategy of improving each of the three dimensions sequentially.
* m2_7_remove_test.py - This Python program removes the test dialogues from the train datasets.
* m2_8_sft_dpo_dataset_generation.py - This Python program converts the CSV datasets to LLaMA-Factory dataset format.

`model-testing`: This folder contains Python programs to generate responses for the fine-tuned models.

`win-rate-evaluation`: This folder contains Python programs to evaluate the responses of the fine-tuned models along the win rate metric, judged by either GPT-4o or Gemini.

## Dependencies
1. You will need to install Python version 3.12.4 on your computer.
2. You will need to import the openai python api package and the genai api package. You can do this by running `pip install openai` and `pip install genai`.
3. This study requires you to have an OpenAI API key and Gemini API key. You will need to store your API key in an environment variable namely `OPENAI_API_KEY` and `GEMINI_API_KEY`.
4. You will need to install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
