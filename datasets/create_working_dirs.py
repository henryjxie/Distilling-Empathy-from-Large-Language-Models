import os

if not os.path.exists("direct-dist"):
    os.makedirs("direct-dist")  

if not os.path.exists("gemini_win_rates"):
    os.makedirs("gemini_win_rates")  

if not os.path.exists("human-impr"):
    os.makedirs("human-impr")  

if not os.path.exists("gpt4o"):
    os.makedirs("gpt4o")  

if not os.path.exists("gpt4o_win_rates"):
    os.makedirs("gpt4o_win_rates")  

if not os.path.exists("llama_3.1_8B"):
    os.makedirs("llama_3.1_8B")  

if not os.path.exists("llm-impr"):
    os.makedirs("llm-impr")  

if not os.path.exists("mistral_7B_v0.3"):
    os.makedirs("mistral_7B_v0.3")  