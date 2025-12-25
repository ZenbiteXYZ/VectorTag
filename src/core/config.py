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
    MODELS_DIR: Optional[Path] = Field(default=None)
    WEIGHTS_NAME: str = "baseline_tagged_30_2.pth"
    CLASSES_NAME: str = "classes_30_2.json"

    # Training Hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 8
    WEIGHT_DECAY: float = 1e-4

    # Dataset Configuration
    TOP_K: int = 30
    FILTER_TO_TOP: bool = True
    MAX_SAMPLES: Optional[int] = 100_000

    # Hardware
    NUM_WORKERS: int = 4

    # Post-Init Logic
    def model_post_init(self, __context):
        if self.DATA_DIR is None:
            self.DATA_DIR = self.PROJECT_ROOT / "data" / "raw" / "various_tagged_images"

        if self.CSV_FILE is None:
            self.CSV_FILE = self.DATA_DIR / "metadata.csv"

        if self.MODELS_DIR is None:
            self.MODELS_DIR = self.PROJECT_ROOT / "models" / "weights"

        # Create dirs
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @computed_field
    def WEIGHTS_PATH(self) -> Path:
        return self.MODELS_DIR / self.WEIGHTS_NAME

    @computed_field
    def CLASSES_PATH(self) -> Path:
        return self.MODELS_DIR / self.CLASSES_NAME

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
