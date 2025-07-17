import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import logging

logger = logging.getLogger(__name__)

class Seq2SeqModel(nn.Module):
    """
    Clase unificada para manejar modelos entrenados para traducción
    usando Helsinki-NLP. Detecta idioma automáticamente y aplica
    LoRA si se indica.
    """

    def __init__(self, mode="full_finetune", lora_config=None):
        """
        Parámetros:
        mode: 'full_finetune' o 'lora'
        """
        super().__init__()
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inicializando modelo en modo: {mode}")

        #USAMOS UN MODELO BASE (LUEGO SE CAMBIA POR IDIOMA)
        self.model_name = "Helsinki-NLP/opus-mt-en-es"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        if self.mode == "lora":
            self._setup_lora(lora_config)

    def _setup_lora(self, lora_config=None):
        logger.info("Aplicando LoRA...")
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=["q_proj", "v_proj"],
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none"
            )
        self.model = get_peft_model(self.model, lora_config)

    def generate(self, input_texts, max_length=128):
        """
        Detecta idioma por texto y selecciona el modelo de traducción adecuado.
        """
        outputs = []
        for text in input_texts:
            #DETECTAR IDIOMA DE FORMA "SIMPLE"
            if any(c in text.lower() for c in "áéíóúñ¿¡"):
                modelo_direccion = "es-en"
                model_name = "Helsinki-NLP/opus-mt-es-en"
            else:
                modelo_direccion = "en-es"
                model_name = "Helsinki-NLP/opus-mt-en-es"

            #CARGAR MODELO ADECUADO SI ES DISTINTO
            if model_name != self.model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                if self.mode == "lora":
                    self._setup_lora()
                self.model_name = model_name

            logger.info(f"Usando modelo: {modelo_direccion} para texto: {text}")

            inputs = self.tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                translated = self.model.generate(**inputs, max_length=max_length)
            decoded = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            outputs.append(decoded[0])
        return outputs