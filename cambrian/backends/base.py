"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract LLM backend.

    All backends must implement ``generate`` which takes a prompt string
    and returns the model's completion as a string.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for *prompt*.

        Args:
            prompt: The user/system prompt to send to the model.
            **kwargs: Backend-specific overrides (temperature, max_tokens, …).

        Returns:
            The model's text completion.

        Raises:
            RuntimeError: If the backend call fails after retries.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier (e.g. ``"gpt-4o"``)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
