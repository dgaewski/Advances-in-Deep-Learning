from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "Assist user with conversion. Be concise. Think step by step and give your final answer inside <answer>number</answer>."},
            {"role": "user", "content": "How much is 1 yd when converted to ft?"},
            {"role": "assistant", "content": "1 yd = 3 ft. 1 * 3 = 3. <answer>3.0</answer>"},
            {"role": "user", "content": "How do we translate 5 gallon into fluid ounce?"},
            {"role": "assistant", "content": "1 gal = 128 fl oz. 5 * 128 = 640. <answer>640.0</answer>"},
            {"role": "user", "content": question}]
        
        
        result = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        #print(result)
        return result

        raise NotImplementedError()


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()

    # # print first 5 answers to see what model outputs
    # for i in range(5):
    #     q, correct = testset[i]
    #     output = model.batched_generate([model.format_prompt(q)])[0]
    #     print(f"Q: {q}")
    #     print(f"Output: {output}")
    #     print(f"Correct: {correct}")
    #     print("---")

    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
