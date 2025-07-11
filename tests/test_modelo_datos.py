import pytest
import torch
from src.modelo_seq2seq import Seq2SeqModel
from src.datos import crear_dataloaders


@pytest.fixture(scope="module")
def modelo():
    return Seq2SeqModel(model_name="t5-small", mode="full_finetune")


@pytest.fixture(scope="module")
def dataloader():
    return crear_dataloaders(batch_size=2)[0]  # solo el train_loader


def test_forward_pass(modelo, dataloader):
    """
    Prueba de paso hacia adelante (forward) del modelo usando un batch real.
    """
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    outputs = modelo(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None
    assert outputs.loss.item() >= 0
    assert outputs.logits.shape[0] == input_ids.shape[0]


def test_generation(modelo):
    """
    Prueba simple de generación de texto traducido.
    """
    entrada = "translate English to Spanish: The book is on the table."
    salida = modelo.generate(entrada)

    assert isinstance(salida, list)
    assert len(salida) == 1
    assert isinstance(salida[0], str)
    assert len(salida[0]) > 0
