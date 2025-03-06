from llm.specific_llm import QwenLLM

if __name__ == "__main__":
    model_path = "models/Qwen2.5-0.5B"
    
    llm = QwenLLM(model_path)  # Can swap with GPTLLM if needed

    data = {
        "prompt": "Which of the stocks available in the S&P 500 would you recommend me to invest?",
        "clear_memory": True,
        "system_message": "You are a financial assistant providing investment advice."
    }

    llm.clear_memory(data["clear_memory"])
    response = llm(data["prompt"], system_message=data["system_message"])

    with open("responses/response.txt", "w") as f:
        f.write(response)
