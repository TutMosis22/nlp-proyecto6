##AQUÍ SE DESCARGARÁ, PROCESARÁ Y PREPARARÁ LOS PARES DE ORACIONES
##EN<->ES PARA ENTRENAR Y EVALUAR NUSTRO MODELO DE TRADUCCIÓN

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import logging

logger = logging.getLogger(__name__)

class TraduccionDataset(Dataset):
    """
    Dataset personalizado para traducción EN <-> ES o ES -> EN usando T5
    Se basa en lo datos de opus_books (https://www.opusebooks.com/)
    """
    def __init__(self, split="train", tokenizer_name="t5-small", max_length=128, direction="en_to_es"):
        """
        split: 'train' o 'test'
        tokenizer_name: nombre del modelo para cargar el tokenizer
        max_length: longitud máxima de entrada y salida
        direction: 'en_to_es' o 'es_to_en'
        """
        super().__init__()
    
        self.max_length = max_length
        self.direction = direction
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        logger.info(f"Cargando el split {split} del dataset opus_books")
        self.dataset = load_dataset("opus_books", "en-es", split=split)
        
        logger.info(f"Total de muestras cargadas: {len(self.dataset)}")
        
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
            
        #TOKENIZACIÓN
        source = self.tokenizer(
            source_text, padding = "max_length", truncation = True, max_length=self.max_length, return_tensors = "pt"     
        )
        
        target = self.tokenizer(
            target_text, padding="max_length", truncation = True, max_length=self.max_length, return_tensors="pt"
        )
        
        return{
            "inputs_ids": source["inputs_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": target["input_ids"].squeeze(0),
        }

def crear_dataloaders(batch_size=8, max_length=128, tokenizer_name="t5-small", direction = "en_to_es"):
    """
    Crea los dataloaders de entrenamiento y prueba
    """
    
    logger.info("Creando dataloaders...")
    
    train_dataset = TraduccionDataset(split = "train", tokenizer_name=tokenizer_name,
                                      max_length = max_length, direction=direction)
    
    test_dataset = TraduccionDataset(split = "test", tokenizer_name = tokenizer_name,
                                    max_length = max_length, direction = direction)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    
    logger.info("Dataloaders listos")
    
    return train_loader, test_loader