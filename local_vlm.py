from llm.specific_llm import QwenVLM

if __name__ == "__main__":
    model_path = "models/Qwen2.5-VL-7B-Instruct"  
    vlm = QwenVLM(model_path)

    text_prompt = "What do you see in this image?"
    image_path = "000000000785.jpg"  # Path to an image file

    response = vlm(text_prompt, image_path=image_path)

    print("VLM Response:", response)
