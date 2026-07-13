from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from datasets import IterableDataset, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoProcessor

DATASET_NAME = "lmms-lab/flickr30k"
DATASET_SPLIT = "test"

MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = "./hf_cache"

TRAIN_SAMPLE_COUNT = 1_000
TEST_SAMPLE_COUNT = 100

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

SHUFFLE_BUFFER_SIZE = 512
MAX_TEXT_LENGTH = 77

SEED = 42

processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

full_stream:IterableDataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
shuffled_stream:IterableDataset = full_stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

train_stream: IterableDataset = shuffled_stream.take(
    TRAIN_SAMPLE_COUNT
)

test_stream: IterableDataset = (
    shuffled_stream
    .skip(TRAIN_SAMPLE_COUNT)
    .take(TEST_SAMPLE_COUNT)
)

class CLIPCollator:
    def __init__(self, processor, max_text_length:int = 77, random_caption:bool = True) -> None:
        self.processor = processor
        self.max_text_length = max_text_length
        self.random_caption = random_caption
    
    def _select_caption(self, captions:str|Sequence[str]) -> str:
        if isinstance(captions, str): return captions
        if self.random_caption: return random.choice(list(captions))
        return captions[0]
    
    def __call__(self, batch:list[dict[str, Any]]) -> dict[str, Tensor]:
        image_list = []
        caption_list = []
        image_id_list = []
        
        for sample in batch:
            image = sample["image"].convert("RGB")
            caption = self._select_caption(sample["caption"])
            
            image_list.append(image)
            caption_list.append(caption)
            image_id_list.append(int(sample["img_id"]))
            
        processed = self.processor(
            images=image_list,
            text=caption_list,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            # [B, 3, 224, 224]
            "pixel_values": processed["pixel_values"],

            # [B, L]
            "input_ids": processed["input_ids"],

            # [B, L]
            "attention_mask": processed["attention_mask"],

            # [B]
            "image_ids": torch.tensor(
                image_id_list,
                dtype=torch.long,
            ),
        }

train_collator = CLIPCollator(processor, MAX_TEXT_LENGTH)
test_collator = CLIPCollator(processor, MAX_TEXT_LENGTH, False)

train_loader = DataLoader(train_stream, TRAIN_BATCH_SIZE, collate_fn=train_collator, num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_stream, TEST_BATCH_SIZE, collate_fn=test_collator, num_workers=0, pin_memory=True, drop_last=False)