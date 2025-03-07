from llm.specific_llm import QwenLLM

if __name__ == "__main__":
    model_path = "models/Qwen2.5-0.5B"
    
    llm = QwenLLM(model_path)  # Can swap with GPTLLM if needed

    print("Welcome to the conversational LLM. Type 'exit' to end the conversation.")
    system_message = "You are a helpful assistant engaging in a conversation."
    
    conversation_active = True
    
    while conversation_active:
        user_input = input("User: ")
        
        if user_input.lower() == "exit":
            conversation_active = False
            print("Ending conversation. Memory cleared.")
            llm.clear_memory(True)  # Clear memory at the end
            break

        response = llm(user_input, system_message=system_message)
        
        print("Assistant:", response)
