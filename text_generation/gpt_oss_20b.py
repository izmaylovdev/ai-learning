"""
GPT-OSS-20B generator implementation using HuggingFace Transformers.
"""

from transformers import pipeline
from typing import Optional
from .generator import GeneratorInterface
import config


class GPTOss20BGenerator(GeneratorInterface):
    """GPT-OSS-20B text generator from HuggingFace using pipeline."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        """
        Initialize the GPT-OSS-20B generator.

        Args:
            model_name: Model name/path (defaults to config.HUGGINGFACE_MODEL)
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name or config.HUGGINGFACE_MODEL
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {self.model_name}")

        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            dtype="auto",
            device_map="auto",
        )

        print(f"Model loaded successfully")

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
        )

        return outputs[0]["generated_text"][-1]["content"]

    def generate_answer(self, question: str, context: str) -> str:

        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        return self.generate(prompt)

