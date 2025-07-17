#PRUEBA RÁPIDA PARA EVALUATION.PY
import sys
import os

from src.evaluation import calcular_bleu, medir_latencia, medir_memoria
from src.modelo_seq2seq import Seq2SeqModel

def test_bleu():
    pred = ["El libro está sobre la mesa."]
    ref = [["El libro está sobre la mesa."]]
    score = calcular_bleu(pred, ref)
    assert score["bleu"] > 0

def test_latencia():
    model = Seq2SeqModel(mode="full_finetune")
    tokenizer = model.tokenizer
    times = medir_latencia(model, tokenizer, ["translate English to Spanish: Hello world"] * 3)
    assert "p95_ms" in times

def test_memoria():
    mem = medir_memoria()
    assert "ram_mb" in mem