# Proyecto 6 - Traducción EN↔ES con LoRA vs Adapters vs Fine-tune Completo

## Descripción

Este proyecto tiene como objetivo comparar tres técnicas de ajuste de parámetros sobre un modelo T5-Small para la tarea de traducción inglés-español:

* Fine-tuning completo del modelo base.
* Ajuste con Adapters (bottleneck).
* Integración de LoRA (Low-Rank Adaptation).

Se evalúa su rendimiento en términos de calidad de traducción (BLEU/COMET), eficiencia (uso de VRAM y tiempo de entrenamiento), y facilidad de implementación.

---

## Estructura del repositorio (de momento)

```
proyecto1_seq2seq/
├── src/                       # Código fuente del proyecto
│   ├── __init__.py
│   ├── modelo_seq2seq.py     # Modelo Encoder-Decoder
│   ├── decoders/             # Decodificadores: greedy, beam, top-k, etc.
│   ├── datos.py              # Carga y procesamiento de datos
│   ├── evaluation.  py         # BLEU, tiempo, memoria
│   
├── tests/                    # Pruebas unitarias y funcionales
│   ├── test_greedy.py
│   ├── test_beam.py
|   ├── test_modelo_datos.py
│   ├── test_evaluation.py
│   
├── benchmarks/               # Scripts y resultados de benchmarks
│   ├── run_bench.sh
│   ├── bench_translation.py
│   ├── results/
│       ├── bleu_vs_latency.csv
│       ├── memoria_vs_len.csv
│
├── exposicion.ipynb          # Cuaderno de exposición final
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
```

---

## Cómo ejecutar

1. Clona el repositorio:

```bash
git clone https://github.com/TutMosis22/nlp-proyecto6.git
cd nlp-proyecto6
```

2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecuta pruebas:

```bash
pytest -q --cov=src
```

4. Corre los benchmarks:

```bash
bash benchmarks/run_bench.sh
```

5. Abre el cuaderno de exposición para visualizar resultados:

```bash
jupyter notebook exposicion.ipynb
```

6. Video demostrativo:
   [Lo colocaré al finalizar el proyecto]

---

## Requisitos

* Python
* torch
* GPU (mínimo 4GB VRAM) o CPU con capacidad de al menos 8GB RAM
* Recomendado: NVIDIA con soporte CUDA para entrenamiento eficiente

---

## Licencia

MIT License

---

## Autor

Andre - Facultad de Ciencias UNI - 2025
