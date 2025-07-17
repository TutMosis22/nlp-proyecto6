# Proyecto 6 - Traducción EN↔ES con LoRA vs Fine-tuning Completo

## Descripción

Este proyecto compara dos enfoques de ajuste de parámetros sobre un modelo T5-Small para traducción inglés ↔ español:

- **Fine-tuning completo**: entrenamiento total del modelo base.
- **LoRA (Low-Rank Adaptation)**: entrenamiento eficiente de parámetros adicionales.

Se evalúan en términos de calidad de traducción (BLEU/ROUGE), eficiencia (latencia y uso de memoria), y se ofrece una **app local** para que el usuario interactúe con ambos modelos y compare los resultados directamente.

---

## Estructura del repositorio


```
NLP-PROYECTO6/
├── app/ # Interfaz web local (Streamlit)
│ └── main.py
├── benchmarks/ # Scripts y notebook para análisis de resultados
│ ├── bench_translation.py
│ ├── benchmark_modelos.ipynb
│ ├── run_bench.sh
│ └── results/
├── checkpoints/ # Modelos entrenados (se generan aquí)
│ ├── full_finetune/
│ └── lora/
├── src/ # Código fuente principal
│ ├── datos.py
│ ├── entrenamiento.py
│ ├── evaluation.py
│ └── modelo_seq2seq.py
├── tests/ # Pruebas unitarias
│ ├── test_evaluation.py
│ └── test_modelo_datos.py
├── experimentos.py # Script que carga modelos, evalúa y compara
├── requirements.txt # Dependencias del proyecto
├── README.md # Este archivo
└── pytest.ini
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

4. Evaluar modelos y comparar métricas:

```bash
python experimentos.py
```
Esto genera comparaciones de BLEU y ROUGE en consola y las guarda en resultados_experimentos.csv

5. Corre los benchmarks (latencia, memoria, etc):

```bash
bash benchmarks/run_bench.sh
```

5. Abre el cuaderno de exposición para visualizar resultados:

```bash
jupyter notebook exposicion.ipynb
```
6. Ejecutar app local (traductor interactivo)

streamlit run app/main.py

7. Video demostrativo:
   [Link en el UNI Virtual]

---

## Requisitos

* Python
* torch
* GPU (mínimo 4GB VRAM) o CPU con capacidad de al menos 8GB RAM
* Recomendado: NVIDIA con soporte CUDA para entrenamiento eficiente

---

## Métricas utilizadas
- BLEU: comparación n-gram entre referencia y predicción
- ROUGE-L: medida basada en subcadenas comunes más largas
- Latencia: p50/p95/p99 (tiempo por traducción)
- Consumo de memoria: RAM y VRAM

Nota: Se descartó el uso de Adapters por simplicidad y compatibilidad.

## Licencia

MIT License

---

## Autor

Andre - Facultad de Ciencias UNI - 2025
