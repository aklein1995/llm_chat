from flask import Flask, request, jsonify
from llm.specific_llm import QwenLLM 

# Initialize Flask app
app = Flask(__name__)

# Load LLM once (Global)
print("[+] Loading model...")
model_path = "models/Qwen2.5-0.5B"
llm = QwenLLM(model_path)
print("[+] Model loaded successfully.")

# Request counter
n_requests = 0

@app.route('/generate', methods=['POST'])
def generate_response():
    """
        Flask endpoint to generate responses from the LLM.
    """
    global n_requests
    n_requests += 1
    print(f"[+] Request received. Request number: {n_requests}")

    # process incoming data
    data = request.get_json()
    prompt = data.get('prompt')
    clear_memory = data.get('clear_memory')
    system_message = data.get('system_message')  # Optional system message

    print('prompt:',prompt)
    print('memoryclear:',clear_memory)
    print('systemmessage:',system_message)

    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    # Clear memory if requested
    if clear_memory:
        print("[+] Clearing memory...")
        llm.clear_memory()

    # Generate prompt
    print("[+] Generating response...")
    response = llm(prompt, system_message=system_message)
    print("[+] Response generated.")
    print('Response:', response)
    
    # Return prompt
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
