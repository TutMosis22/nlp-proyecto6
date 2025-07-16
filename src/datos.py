from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import logging

logger = logging.getLogger(__name__)

class TraduccionDataset(Dataset):
    """
    Dataset personalizado para traducción EN <-> ES usando T5
    """
    def __init__(self, dataset_split, tokenizer_name="t5-small", max_length=128, direction="en_to_es"):
        """
        dataset_split: un objeto Hugging Face Dataset (ya dividido como train o test)
        tokenizer_name: nombre del modelo para cargar el tokenizer
        max_length: longitud máxima de entrada y salida
        direction: 'en_to_es' o 'es_to_en'
        """
        super().__init__()

        self.dataset = dataset_split
        self.max_length = max_length
        self.direction = direction
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.direction == "en_to_es":
            source_text = "translate English to Spanish: " + item["translation"]["en"]
            target_text = item["translation"]["es"]
        else:
            source_text = "translate Spanish to English: " + item["translation"]["es"]
            target_text = item["translation"]["en"]

        source = self.tokenizer(
            source_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        target = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": target["input_ids"].squeeze(0)
        }


def crear_dataloaders(batch_size=8, max_length=128, tokenizer_name="t5-small", direction="en_to_es"):
    """
    Crea los dataloaders de entrenamiento y validación a partir de un solo split 'train'
    """
    logger.info("Cargando y dividiendo dataset opus_books...")

    dataset = load_dataset("opus_books", "en-es", split="train")
    train_split = dataset.select(range(int(0.9 * len(dataset))))
    test_split = dataset.select(range(int(0.9 * len(dataset)), len(dataset)))

    train_dataset = TraduccionDataset(train_split, tokenizer_name=tokenizer_name,
                                      max_length=max_length, direction=direction)
    test_dataset = TraduccionDataset(test_split, tokenizer_name=tokenizer_name,
                                     max_length=max_length, direction=direction)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logger.info("Dataloaders listos.")
    return train_loader, test_loader