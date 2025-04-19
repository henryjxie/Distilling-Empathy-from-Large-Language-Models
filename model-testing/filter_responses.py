import pandas as pd
import importlib.util

# Prompt the user to enter the path to the Python configuration file
config_path = input("Enter the path to the Python configuration file for the model: ")

# Dynamically import the configuration module
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

response_file = pd.read_csv(config.OUTPUT_PATH)

response_file["response"] = response_file["response"].str.split("Response: ", n=1).str[-1]

response_file.to_csv(config.OUTPUT_PATH_FILTERED, index=False)