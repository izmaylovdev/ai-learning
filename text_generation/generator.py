from abc import ABC, abstractmethod

class GeneratorInterface(ABC):
    """Interface for text generation models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the prompt.

        Args:
            prompt: Input prompt for generation

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer to a question given context.

        Args:
            question: User's question
            context: Retrieved context from documents

        Returns:
            Generated answer
        """
        pass
