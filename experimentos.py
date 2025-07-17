from src.modelo_seq2seq import Seq2SeqModel
from src.datos import crear_dataloaders
from src.evaluation import calcular_bleu, calcular_rouge
import pandas as pd

def evaluar_modelo(nombre, modo):
    print(f"\nEvaluando modelo: {nombre}")
    modelo = Seq2SeqModel(model_name="t5-small", mode=modo)
    _, test_loader = crear_dataloaders(batch_size=4)
    batch = next(iter(test_loader))
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    textos_entrada = modelo.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    textos_referencia = modelo.tokenizer.batch_decode(labels, skip_special_tokens=True)
    textos_generados = modelo.generate(textos_entrada)
    
    bleu = calcular_bleu(textos_generados, [[ref] for ref in textos_referencia])
    rouge = calcular_rouge(textos_generados, textos_referencia)
    
    for i in range(len(textos_entrada)):
        print(f"\nEntrada     : {textos_entrada[i]}")
        print(f"Referencia  : {textos_referencia[i]}")
        print(f"Traducción  : {textos_generados[i]}")
    
    return {
        "modelo": nombre,
        "BLEU": round(bleu["bleu"], 4),
        "ROUGE-L": round(rouge["rougeL"], 4)
    }

def main():
    resultados = []
    resultados.append(evaluar_modelo("Fine-tuning completo", "full_finetune"))
    resultados.append(evaluar_modelo("LoRA", "lora"))
    
    df = pd.DataFrame(resultados)
    print("\nResumen de comparación:\n", df)
    df.to_csv("resultados_experimentos.csv", index=False)

if __name__ == "__main__":
    main()