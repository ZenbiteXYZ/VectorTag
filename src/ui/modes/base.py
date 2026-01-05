from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
from PIL import Image


class InferenceMode(ABC):
    """Base class for different tagging stratagies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the mode."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the mode."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image, threshold: float) -> List[Tuple[str, float]]:
        """
        Returns a list of (tag, probability) tuples.
        """
        raise NotImplementedError

    def get_available_models(self) -> Dict[str, str]:
        """
        Returns a dictionary of {display_name: model_identifier}.
        Example: {"Baseline v1": "baseline_v1", "Baseline v2": "baseline_v2"}

        If mode doesn't support multiple models, return empty dict or single entry.
        """
        return {}

    def set_model(self, model_key: str):
        """
        Switch to a different model variant.

        Args:
            model_key (str): One of the keys from get_available_models().
        """
        raise NotImplementedError("This mode does not support model switching.")

class TagManagementMixin(ABC):
    """Mixin for modes that support dynamic tag management (add/remove)."""

    @abstractmethod
    def get_all_tags(self) -> List[str]:
        """Returns a list of all available tags in the database."""
        raise NotImplementedError

    @abstractmethod
    def add_tag(self, tag: str, description: Optional[str] = None):
        """Adds a new tag to the semantic database."""
        raise NotImplementedError

    @abstractmethod
    def remove_tag(self, tag: str):
        """Removes a tag from the semantic database."""
        raise NotImplementedError
