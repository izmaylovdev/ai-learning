from text_generation.generator import GeneratorInterface

def get_generator(generator_type: str = "huggingface", **kwargs) -> GeneratorInterface:
    if generator_type.lower() == "huggingface":
        from text_generation.gpt_oss_20b import GPTOss20BGenerator
        return GPTOss20BGenerator(**kwargs)
    elif generator_type.lower() == "gemini":
        # Import here to avoid circular dependency
        try:
            from text_generation.google_gemini import GeminiGenerator
            return GeminiGenerator(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"Could not import GeminiGenerator. Make sure google-genai is installed. Error: {e}"
            )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

