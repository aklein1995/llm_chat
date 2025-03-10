from llm.specific_llm import QwenVLM

if __name__ == "__main__":
    model_path = "models/Qwen2.5-VL-7B-Instruct"  
    vlm = QwenVLM(model_path)

    text_prompt = "What do you see in these 3 images? Please provide a description for each image"
    image_paths = [
        "images/000000000785.jpg",
        "images/000000000139.jpg",
        "images/000000000285.jpg"
    ]

    response = vlm(text_prompt, image_paths)

    print("\nVLM Response:", response)
