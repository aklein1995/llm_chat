import requests

# Define the Flask API endpoint
API_URL = "http://localhost:8000/generate"

# Define the request payload
data = {
    "prompt": "Which stocks from the S&P 500 are worth investing in?",
    "clear_memory": True,
    "system_message": "You are a financial advisor. Only provide factual information."
}

# Send the request
response = requests.post(API_URL, json=data)

# Print the response
if response.status_code == 200:
    answer = response.json()["response"]
    print("[+] Response from LLM:")
    print(answer)
    
    with open("responses/response.txt", "w") as f:
        f.write(answer)
else:
    print(f"[!] Error: {response.status_code}")
    print(response.json())
