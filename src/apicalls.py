import requests
import os
from datetime import datetime


# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# Call each API endpoint and store the responses
response1 = requests.post(f"{URL}prediction")
response2 = requests.get(f"{URL}scoring")
response3 = requests.get(f"{URL}summarystats")
response4 = requests.get(f"{URL}diagnostics")

# Combine all API responses
responses = [response1.json(), response2.json(), response3.json(), response4.json()]

# Write the responses to a file in your workspace
path2api_responses = os.path.join("..", "api_reponses")
if not os.path.exists(path2api_responses):
    os.makedirs(path2api_responses)

# Get the current date and time and format it as a string
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Append the timestamp to the output file name
output_file_name = f"api_responses_{current_timestamp}.txt"
output_file_path = os.path.join(path2api_responses, output_file_name)

with open(output_file_path, "w") as f:
    for response in responses:
        f.write(str(response) + "\n")
