import torch
import time
import numpy as np
import evaluate
import psutil

from transformers import AutoTokenizer

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


#MÉTRICAS DE CALIDAD

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def calcular_bleu(predicciones, referencias):
    """
    Calcula BLEU score.
    """
    return bleu_metric.compute(predictions=predicciones, references=referencias)


def calcular_rouge(predicciones, referencias):
    """
    Calcula ROUGE-L.
    """
    return rouge_metric.compute(predictions=predicciones, references=referencias)


#MÉTRICAS DE RENDIMIENTO

def medir_latencia(modelo, tokenizer, textos, max_length=128):
    """
    Mide tiempo de inferencia (latencia) sobre una lista de textos.
    Devuelve P50, P95, P99 en milisegundos.
    """
    modelo.model.eval()
    tiempos = []

    for text in textos:
        inputs = tokenizer(text, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            _ = modelo.model.generate(**inputs, max_length=max_length)
        end = time.time()
        tiempos.append((end - start) * 1000)  # ms

    tiempos = np.array(tiempos)
    return {
        "p50_ms": np.percentile(tiempos, 50),
        "p95_ms": np.percentile(tiempos, 95),
        "p99_ms": np.percentile(tiempos, 99),
    }


def medir_memoria():
    """
    Mide uso de memoria del sistema y GPU (si está disponible).
    """
    memoria_ram = psutil.virtual_memory().used / (1024 ** 2)  # MB

    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memoria_vram = info.used / (1024 ** 2)  # MB
    else:
        memoria_vram = None

    return {
        "ram_mb": round(memoria_ram, 2),
        "vram_mb": round(memoria_vram, 2) if memoria_vram else "N/A"
    }