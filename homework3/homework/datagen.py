import json
import os
import re   #regex for parsing the <answer> tags
from .cot import CoTModel
from .sft import load, train_model, tokenize, format_example
from .data import Dataset, benchmark


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    
    #load training dataset
    print("Loading model...")
    model = CoTModel(checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct")
    print(f"Using device: {model.device}") # Check if this says 'cuda'
    train_data = Dataset("train")
    rft_samples = []
    print("Model loaded")

    # Process in batches of questions
    batch_size = 4
    for start in range(0, len(train_data), batch_size):
        print(f"Starting batch {start // batch_size + 1}...")

        batch = [train_data[i] for i in range(start, min(start + batch_size, len(train_data)))]
        prompts = [model.format_prompt(q) for q, a in batch]

        all_completions = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)
        print(f"Batch done.")


        for j, (question, correct_answer) in enumerate(batch):
            success_found = False
            for reasoning in all_completions[j]:  # <-- this loop is missing
                match = re.search(r"<answer>(.*?)</answer>", reasoning)
                if match:
                    try:
                        val_str = match.group(1).strip().replace(',', '')
                        predicted_val = float(val_str)
                        if abs(predicted_val - correct_answer) < abs(correct_answer) * 0.01 + 1e-4:
                            rft_samples.append([question, correct_answer, reasoning])
                            success_found = True
                            break
                    except ValueError:
                        continue

        # If success_found is False, the question is ignored (rejection sampling)
        # Progress tracking
        status = "✓" if success_found else "✗"
        print(f"[{min(start + batch_size, len(train_data))}/{len(train_data)}] | Samples so far: {len(rft_samples)}")

    #Save the successful rollouts to the json file
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(rft_samples, f, indent=4)
    
    print(f"RFT dataset created with {len(rft_samples)} samples.")


    #raise NotImplementedError()


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
