from openai import OpenAI
import os
import json

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Naive prompt - SFT
naive_prompt_sft_batch_input_file = client.files.create(
    file=open("datasets/naive_prompt_human_sft_dataset.jsonl", "rb"),
    purpose="batch"
)

naive_prompt_sft_batch_object = client.batches.create(
    input_file_id=naive_prompt_sft_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Naive Prompt Human SFT Batch"
    }
)

naive_prompt_sft_batch_object = naive_prompt_sft_batch_object.to_dict()

with open("datasets/naive_prompt_sft_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(naive_prompt_sft_batch_object, file, indent=2)

# Instruct to improve on all three dimensions of empathy - SFT
improve_all_three_dimensions_sft_batch_input_file = client.files.create(
    file=open("datasets/improve_all_three_dimensions_human_sft_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_all_three_dimensions_sft_batch_object = client.batches.create(
    input_file_id=improve_all_three_dimensions_sft_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve On All Three Dimensions Of Empathy Human SFT Batch"
    }
)

improve_all_three_dimensions_sft_batch_object = improve_all_three_dimensions_sft_batch_object.to_dict()

with open("datasets/improve_all_three_dimensions_sft_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_all_three_dimensions_sft_batch_object, file, indent=2)

# Instruct to identify which dimensions of empathy the original response is lacking the most in, and improve along that dimension of empathy - SFT
identify_lacking_dimension_sft_batch_input_file = client.files.create(
    file=open("datasets/identify_lacking_dimension_human_sft_dataset.jsonl", "rb"),
    purpose="batch"
)

identify_lacking_dimension_sft_batch_object = client.batches.create(
    input_file_id=identify_lacking_dimension_sft_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Identify Lacking Dimensions Human SFT Batch"
    }
)

identify_lacking_dimension_sft_batch_object = identify_lacking_dimension_sft_batch_object.to_dict()

with open("datasets/identify_lacking_dimension_sft_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(identify_lacking_dimension_sft_batch_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (cognitive) - SFT
improve_cognitive_sft_batch_input_file = client.files.create(
    file=open("datasets/improve_cognitive_human_sft_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_cognitive_sft_batch_object = client.batches.create(
    input_file_id=improve_cognitive_sft_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Cognitive Human SFT Batch"
    }
)

improve_cognitive_sft_batch_object = improve_cognitive_sft_batch_object.to_dict()

with open("datasets/improve_cognitive_sft_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_cognitive_sft_batch_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (affective) - SFT
improve_affective_sft_batch_input_file = client.files.create(
    file=open("datasets/improve_affective_human_sft_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_affective_sft_batch_object = client.batches.create(
    input_file_id=improve_affective_sft_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Affective Human SFT Batch"
    }
)

improve_affective_sft_batch_object = improve_affective_sft_batch_object.to_dict()

with open("datasets/improve_affective_sft_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_affective_sft_batch_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (compassionate) - SFT
improve_compassionate_sft_batch_input_file = client.files.create(
    file=open("datasets/improve_compassionate_human_sft_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_compassionate_sft_batch_object = client.batches.create(
    input_file_id=improve_compassionate_sft_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Compassionate Human SFT Batch"
    }
)

improve_compassionate_sft_batch_object = improve_compassionate_sft_batch_object.to_dict()

with open("datasets/improve_compassionate_sft_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_compassionate_sft_batch_object, file, indent=2)

# Naive prompt - RLHF
naive_prompt_rlhf_batch_input_file = client.files.create(
    file=open("datasets/naive_prompt_human_rlhf_dataset.jsonl", "rb"),
    purpose="batch"
)

naive_prompt_rlhf_batch_object = client.batches.create(
    input_file_id=naive_prompt_rlhf_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Naive Prompt Human RLHF Batch"
    }
)

naive_prompt_rlhf_batch_object = naive_prompt_rlhf_batch_object.to_dict()

with open("datasets/naive_prompt_rlhf_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(naive_prompt_rlhf_batch_object, file, indent=2)

# Instruct to improve on all three dimensions of empathy - RLHF
improve_all_three_dimensions_rlhf_batch_input_file = client.files.create(
    file=open("datasets/improve_all_three_dimensions_human_rlhf_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_all_three_dimensions_rlhf_batch_object = client.batches.create(
    input_file_id=improve_all_three_dimensions_rlhf_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve On All Three Dimensions Of Empathy Human RLHF Batch"
    }
)

improve_all_three_dimensions_rlhf_batch_object = improve_all_three_dimensions_rlhf_batch_object.to_dict()

with open("datasets/improve_all_three_dimensions_rlhf_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_all_three_dimensions_rlhf_batch_object, file, indent=2)

# Instruct to identify which dimensions of empathy the original response is lacking the most in, and improve along that dimension of empathy - RLHF
identify_lacking_dimension_rlhf_batch_input_file = client.files.create(
    file=open("datasets/identify_lacking_dimension_human_rlhf_dataset.jsonl", "rb"),
    purpose="batch"
)

identify_lacking_dimension_rlhf_batch_object = client.batches.create(
    input_file_id=identify_lacking_dimension_rlhf_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Identify Lacking Dimensions Human RLHF Batch"
    }
)

identify_lacking_dimension_rlhf_batch_object = identify_lacking_dimension_rlhf_batch_object.to_dict()

with open("datasets/identify_lacking_dimension_rlhf_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(identify_lacking_dimension_rlhf_batch_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (cognitive) - RLHF
improve_cognitive_rlhf_batch_input_file = client.files.create(
    file=open("datasets/improve_cognitive_human_rlhf_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_cognitive_rlhf_batch_object = client.batches.create(
    input_file_id=improve_cognitive_rlhf_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Cognitive Human RLHF Batch"
    }
)

improve_cognitive_rlhf_batch_object = improve_cognitive_rlhf_batch_object.to_dict()

with open("datasets/improve_cognitive_rlhf_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_cognitive_rlhf_batch_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (affective) - RLHF
improve_affective_rlhf_batch_input_file = client.files.create(
    file=open("datasets/improve_affective_human_rlhf_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_affective_rlhf_batch_object = client.batches.create(
    input_file_id=improve_affective_rlhf_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Affective Human RLHF Batch"
    }
)

improve_affective_rlhf_batch_object = improve_affective_rlhf_batch_object.to_dict()

with open("datasets/improve_affective_rlhf_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_affective_rlhf_batch_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (compassionate) - RLHF
improve_compassionate_rlhf_batch_input_file = client.files.create(
    file=open("datasets/improve_compassionate_human_rlhf_dataset.jsonl", "rb"),
    purpose="batch"
)

improve_compassionate_rlhf_batch_object = client.batches.create(
    input_file_id=improve_compassionate_rlhf_batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Compassionate Human RLHF Batch"
    }
)

improve_compassionate_rlhf_batch_object = improve_compassionate_rlhf_batch_object.to_dict()

with open("datasets/improve_compassionate_rlhf_batch_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_compassionate_rlhf_batch_object, file, indent=2)