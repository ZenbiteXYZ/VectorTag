import os
import torch
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter
from typing import List, Set, Optional, Callable
from src.data.taxonomy import SYNONYMS, IMPLICATIONS, IGNORE_TAGS


class TaggedImagesDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        csv_file_path: str,
        transform=Optional[Callable],
        top_k: int = 30,
        classes: Optional[List[str]] = None,
        filter_to_top: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root_dir (str): Directory with all the images.
            csv_file (str): Path to metadata.csv file.
            transform (Callable, optional): Optional transform to be applied on a sample.
            top_k (int): Number of top frequent tags to use if classes in None.
            classes (List[str], optional): List of class names. If None, will be inferred from filenames.
            filter_to_top (bool): Optional receive rows that contain at least one of the selected classes.
            max_samples (int, optional): Optional value of max samples to receive.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file_path)

        # Remove rows with missing filenames
        self.df = self.df[self.df['image_path'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))]

        # Parse tags and apply logic immediately
        self.df['parsed_tags'] = self.df['tags'].apply(self._parse_and_enrich_tags)

        # Determine the working class set (top-k by frequency or provided)
        if classes is None:
            self.classes = self._get_top_classes(top_k)
        else:
            self.classes = classes

        # Optionally filter dataset rows to only those that contain at least one of the selected classes
        if filter_to_top:
            before = len(self.df)
            class_set = set(self.classes)
            self.df = self.df[self.df['parsed_tags'].apply(lambda tags: any(t in class_set for t in tags))]
            self.df = self.df.reset_index(drop=True)
            after = len(self.df)
            print(f"Filtered to top classes: {before} -> {after} samples")

        # Optionally subsample to a maximum number of samples
        if max_samples is not None and len(self.df) > max_samples:
            print(f"Applying smart balancing to reduce {len(self.df)} -> {max_samples}...")
            self.df = self._smart_subsample(self.df, max_samples, class_set)
            print(f"Subsampled dataset to {len(self.df)} samples.")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"Found {len(self.classes)} unique classes.")

    def _smart_subsample(self, df: pd.DataFrame, n_samples: int, target_classes: Set[str]) -> pd.DataFrame:
        """
        Performs weighted sampling based on the inverse frequency of the rarest tag in each image.
        This ensures rare classes are preserved while majority classes are downsampled.

        Args:
            df (pd.DataFrame): Dataset to apply subsampling to.
            n_samples (int): Number of samples to get.
            target_classes (Set[str]): Required subsample classes.
        """
        # 1. Count frequencies of target classes in the current filtered dataframe
        all_tags: List[str] = []
        for tags in df['parsed_tags']:
            # Only count tags that are in our target list
            all_tags.extend([t for t in tags if t in target_classes])

        counts = Counter(all_tags)

        # 2. Calculate weight for each image
        weights = []
        for tags in df['parsed_tags']:
            # Filter tags relevant to our task
            relevant_tags = [t for t in tags if t in target_classes]

            if not relevant_tags:
                weights.append(0)
                continue

            # Find the count of the rarest tag in this image
            # Example: Image has ['person' (1000), 'horse' (10)]. Min count is 10.
            min_count = min(counts[t] for t in relevant_tags)

            # Weight is inverse frequency. Add small epsilon to avoid div by zero.
            weight = 1.0 / (min_count + 1e-5)
            weights.append(weight)

        # 3. Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 4. Sample
        return df.sample(n=n_samples, weights=weights, random_state=42).reset_index(drop=True)

    def _parse_and_enrich_tags(self, x: str):
        """Parses string, handles synonyms, and adds implied parent tags."""
        if not isinstance(x, str):
            return []

        try:
            # Handle both list strings "['a', 'b']" and simple strings "a, b"
            if x.strip().startswith("["):
                raw_tags = ast.literal_eval(x)
            else:
                raw_tags = [t.strip() for t in x.split(',')]
        except:
            return []

        final_tags = set()

        for tag in raw_tags:
            tag = tag.lower().strip()

            if tag in IGNORE_TAGS:
                continue

            # 1. Resolve Synonyms
            current_tag = SYNONYMS.get(tag, tag)
            final_tags.add(current_tag)

            # 2. Resolve Implications (Recursive)
            # Example: tree -> plant -> nature
            temp_tag = current_tag
            while temp_tag in IMPLICATIONS:
                parent_tag = IMPLICATIONS[temp_tag]

                # Avoid infinite loops if taxonomy has cycles
                if parent_tag in final_tags:
                    break

                final_tags.add(parent_tag)
                temp_tag = parent_tag

        return list(final_tags)

    def _get_top_classes(self, k: int) -> List[str]:
        all_tags: List[str] = []
        for tags in self.df['parsed_tags']:
            all_tags.extend(tags)

        # Count tag frequencies
        counter = Counter(all_tags)
        top_tags = [tag for tag, count in counter.most_common(k)]

        print("\n=== Top 30 Tags Analysis ===")
        for tag, count in counter.most_common(k):
            print(f"{tag}: {count}")
        print("============================\n")

        return top_tags

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('RGB')

        tags = self.df.iloc[idx]['parsed_tags']

        # Create multi-hot label vector
        label_vector = torch.zeros(len(self.classes), dtype=torch.float32)
        for tag in tags:
            if tag in self.class_to_idx:
                label_vector[self.class_to_idx[tag]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label_vector
