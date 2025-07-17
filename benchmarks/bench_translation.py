import os
import csv
from src.modelo_seq2seq import Seq2SeqModel
from src.datos import crear_dataloaders
from src.evaluation import calcular_bleu, calcular_rouge, medir_latencia, medir_memoria

import torch

RESULTS_DIR = "benchmarks/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluar_modelo(modo):
    print(f"Evaluando modo: {modo}")

    modelo = Seq2SeqModel(model_name="t5-small", mode=modo, load_pretrained = True)
    tokenizer = modelo.tokenizer

    # LIMITAR DATOS PARA BENCHMARKING RÁPIDO
    _, test_loader = crear_dataloaders(batch_size=2)
    batch = next(iter(test_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # INFERENCIA
    textos_entrada = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    textos_referencia = tokenizer.batch_decode(labels, skip_special_tokens=True)
    textos_generados = modelo.generate(textos_entrada)

    #MÉTRICAS DE CALIDAD
    bleu = calcular_bleu(textos_generados, [[ref] for ref in textos_referencia])
    rouge = calcular_rouge(textos_generados, textos_referencia)

    # LATENCIA
    latencias = medir_latencia(modelo, tokenizer, textos_entrada)

    #MEMORIA
    memoria = medir_memoria()

    return {
        "modo": modo,
        "bleu": round(bleu["bleu"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "p50_ms": round(latencias["p50_ms"], 2),
        "p95_ms": round(latencias["p95_ms"], 2),
        "p99_ms": round(latencias["p99_ms"], 2),
        "ram_mb": memoria["ram_mb"],
        "vram_mb": memoria["vram_mb"]
    }

def main():
    #modos = ["full_finetune", "adapters", "lora"]
    modos = ["full_finetune", "lora"]

    resultados = []

    for modo in modos:
        resultados.append(evaluar_modelo(modo))

    #GUARDAMOS CSV
    path_csv = os.path.join(RESULTS_DIR, "resumen_benchmark.csv")
    with open(path_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

    print(f"Resultados guardados en {path_csv}")


if __name__ == "__main__":
    main()