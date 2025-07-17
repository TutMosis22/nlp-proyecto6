import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import os

from src.modelo_seq2seq import Seq2SeqModel
from src.datos import crear_dataloaders


def entrenar_modelo(modo, output_dir, device):
    print(f"\nEntrenando modelo en modo: {modo}")

    # CREAR DATALOADERS
    train_loader, _ = crear_dataloaders(batch_size=8, direction="en_to_es")

    #INICIALIZAR MODELO
    modelo = Seq2SeqModel(model_name="t5-small", mode=modo).to(device)

    #OPTIMIZER
    optimizer = AdamW(modelo.parameters(), lr=5e-5)

    epochs = 3
    modelo.train()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = modelo(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nPérdida promedio: {avg_loss:.4f}")

    #CREAR DIRECTORIO Y GUARDAR MODELO Y TOKENIZER
    os.makedirs(output_dir, exist_ok=True)
    modelo.model.save_pretrained(output_dir)
    modelo.tokenizer.save_pretrained(output_dir)
    print(f"\nModelo guardado en: {output_dir}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ENTRENAMIENTO full finetune
    entrenar_modelo("full_finetune", "checkpoints/full_finetune", device)

    # ENTRENAMIENTO LoRA
    entrenar_modelo("lora", "checkpoints/lora", device)