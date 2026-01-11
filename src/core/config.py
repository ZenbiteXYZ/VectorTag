import torch
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class Settings(BaseSettings):
    # Project Paths
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.absolute()
    )

    DATA_DIR: Optional[Path] = Field(default=None, description="Path to raw images directory")
    CSV_FILE: Optional[Path] = Field(default=None, description="Path to metadata.csv")
    MODELS_ROOT: Optional[Path] = Field(default=None)
    STANDARD_WEIGHTS_DIR: Optional[Path] = Field(default=None)
    STANDARD_CLASSES_DIR: Optional[Path] = Field(default=None)

    DEFAULT_STANDARD_MODEL: str = "baseline_v5"

    # Training Hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 12
    WEIGHT_DECAY: float = 1e-5

    # Dataset Configuration
    TOP_K: int = 150
    FILTER_TO_TOP: bool = True
    MAX_SAMPLES: Optional[int] = 200_000

    # Hardware
    NUM_WORKERS: int = 8

    # Post-Init Logic
    def model_post_init(self, __context):
        # Data paths
        if self.DATA_DIR is None:
            self.DATA_DIR = self.PROJECT_ROOT / "data" / "raw" / "various_tagged_images"
        if self.CSV_FILE is None:
            self.CSV_FILE = self.DATA_DIR / "metadata.csv"

        # Models root
        if self.MODELS_ROOT is None:
            self.MODELS_ROOT = self.PROJECT_ROOT / "models"

        # Standard model paths
        if self.STANDARD_WEIGHTS_DIR is None:
            self.STANDARD_WEIGHTS_DIR = self.MODELS_ROOT / "standard" / "weights"
        if self.STANDARD_CLASSES_DIR is None:
            self.STANDARD_CLASSES_DIR = self.MODELS_ROOT / "standard" / "classes"

        # Create directories
        self.STANDARD_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self.STANDARD_CLASSES_DIR.mkdir(parents=True, exist_ok=True)

    @computed_field
    def DEFAULT_WEIGHTS_PATH(self) -> Path:
        return self.STANDARD_WEIGHTS_DIR / f"{self.DEFAULT_STANDARD_MODEL}.safetensors"

    @computed_field
    def DEFAULT_CLASSES_PATH(self) -> Path:
        return self.STANDARD_CLASSES_DIR / f"{self.DEFAULT_STANDARD_MODEL}.json"

    @property
    def DEVICE(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pydantic Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance
settings = Settings()
