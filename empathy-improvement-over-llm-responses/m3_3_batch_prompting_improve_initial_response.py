from openai import OpenAI
import os
import json

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Naive prompt - SFT
naive_prompt_input_file = client.files.create(
    file=open("../datasets/llm-impr/naive_prompt_improve_initial_response.jsonl", "rb"),
    purpose="batch"
)

naive_prompt_object = client.batches.create(
    input_file_id=naive_prompt_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Naive Prompt Improve Initial Response"
    }
)

naive_prompt_object = naive_prompt_object.to_dict()

with open("../datasets/llm-impr/naive_prompt_object.json", "w", encoding="utf-8") as file:
    json.dump(naive_prompt_object, file, indent=2)

# Instruct to improve on all three dimensions of empathy - SFT
improve_all_three_dimensions_input_file = client.files.create(
    file=open("../datasets/llm-impr/improve_all_three_dimensions_improve_initial_response.jsonl", "rb"),
    purpose="batch"
)

improve_all_three_dimensions_object = client.batches.create(
    input_file_id=improve_all_three_dimensions_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve On All Three Dimensions Of Empathy Improve Initial Response"
    }
)

improve_all_three_dimensions_object = improve_all_three_dimensions_object.to_dict()

with open("../datasets/llm-impr/improve_all_three_dimensions_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_all_three_dimensions_object, file, indent=2)

# Instruct to identify which dimensions of empathy the original response is lacking the most in, and improve along that dimension of empathy - SFT
identify_lacking_dimension_input_file = client.files.create(
    file=open("../datasets/llm-impr/identify_lacking_dimension_improve_initial_response.jsonl", "rb"),
    purpose="batch"
)

identify_lacking_dimension_object = client.batches.create(
    input_file_id=identify_lacking_dimension_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Identify Lacking Dimensions Improve Initial Response"
    }
)

identify_lacking_dimension_object = identify_lacking_dimension_object.to_dict()

with open("../datasets/llm-impr/identify_lacking_dimension_object.json", "w", encoding="utf-8") as file:
    json.dump(identify_lacking_dimension_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (cognitive) - SFT
improve_cognitive_input_file = client.files.create(
    file=open("../datasets/llm-impr/improve_cognitive_improve_initial_response.jsonl", "rb"),
    purpose="batch"
)

improve_cognitive_object = client.batches.create(
    input_file_id=improve_cognitive_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Cognitive Improve Initial Response"
    }
)

improve_cognitive_object = improve_cognitive_object.to_dict()

with open("../datasets/llm-impr/improve_cognitive_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_cognitive_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (affective) - SFT
improve_affective_input_file = client.files.create(
    file=open("../datasets/llm-impr/improve_affective_improve_initial_response.jsonl", "rb"),
    purpose="batch"
)

improve_affective_object = client.batches.create(
    input_file_id=improve_affective_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Affective Improve Initial Response"
    }
)

improve_affective_object = improve_affective_object.to_dict()

with open("../datasets/llm-impr/improve_affective_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_affective_object, file, indent=2)

# Instruct to improve on one of the three dimensions only (compassionate) - SFT
improve_compassionate_input_file = client.files.create(
    file=open("../datasets/llm-impr/improve_compassionate_improve_initial_response.jsonl", "rb"),
    purpose="batch"
)

improve_compassionate_object = client.batches.create(
    input_file_id=improve_compassionate_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Improve Compassionate Improve Initial Response"
    }
)

improve_compassionate_object = improve_compassionate_object.to_dict()

with open("../datasets/llm-impr/improve_compassionate_object.json", "w", encoding="utf-8") as file:
    json.dump(improve_compassionate_object, file, indent=2)