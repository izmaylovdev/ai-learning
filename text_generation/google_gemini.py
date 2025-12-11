from google import genai
from typing import Optional
from text_generation.generator import GeneratorInterface
import config


class GeminiGenerator(GeneratorInterface):
    """Google Gemini-based text generator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None
    ):
        self.generation_config = {
            "temperature": temperature or config.GEMINI_TEMPERATURE,
            "top_p": top_p or config.GEMINI_TOP_P,
            "top_k": top_k or config.GEMINI_TOP_K,
            "max_output_tokens": max_output_tokens or config.GEMINI_MAX_OUTPUT_TOKENS,
        }

        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass it to the constructor."
            )

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config={
                    'temperature': self.generation_config["temperature"],
                    'top_p': self.generation_config["top_p"],
                    'top_k': self.generation_config["top_k"],
                }
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating content with Gemini: {str(e)}")

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        return self.generate(prompt)
