import torch
from torch import nn
from transformers import T5ForConditionalGeneration, AutoTokenizer

from peft import get_peft_model, LoraConfig, TaskType
from transformers.adapters import AdapterConfig

import logging

logger = logging.getLogger(__name__)

class Seq2SeqModel(nn.Module):
    """
    Modelo Seq2Seq basado en T5-Small con tres modos de entrenaminto:
    - Fine-tuning
    - Adapters
    - LoRA
    """
    
    def __init__(self, model_name="t5-small", mode="full_finetune", adapter_config=None, lora_config=None):
        """
        Inicialización del modelo-
        
        Parámetros:
        model_name: nombre del modelo preentrenado
        mode: 'full_finetune', 'adapters' o 'lroa'
        adapter_config: configuración opcional para adapters
        lora_config: configuración opcional para LoRA
        """
        super().__init__()
        
        self.mode = mode
        self.model_name = model_name
        logger.info(f"Inicializando modelo T5 con modo: {mode}")
        
        #CARGAMOS EL MODELO Y TOKENIZER
        self.tokenizer = AutoTokenizer.from_pretained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if self.mode == "full_finetune":
            self._setup_full_finetune()
        elif self.mode == "adapters":
            self._setup_adapters(adapter_config)
        elif self.mode == "lora":
            self._setup_lora(lora_config)
        else:
            raise ValueError(f"Modo de entrenamiento desconocido: {self.mode}")
    
    def _setup_full_finetune(self):
        """
        Modo por defecto: se entrena tod el modelo
        """
        logger.info("Fine-tuning completo: todos los parámetros entrenables")
        for param in self.model.parameters():
            param.requires_grad = True
            
    def _setup_adapters(self, adapter_config = None):
        """
        Activa Adapters en el modelo
        Usa la API de Hugging Face transformers.adapters
        """
        logger.info("Configurando Adapters...")
        if adapter_config in None:
            adapter_config = AdapterConfig.load("pfeiffer", reduction_factor = 16)
        self.model.add_adapter("en_es_adapter", config=adapter_config)
        self.model.train_adapter("en_es_adapter")
        
    def _setup_lora(self, lora_config = None):
        """
        Aplica LoRA usando PEFT (Parameter-Efficient Fine Tuning)
        """
        logger.info("Configurando LoRA...")
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r = 8,
                lora_alpha = 16,
                lora_dropout = 0.1,
                bias = "None"
            )
        self.model = get_peft_model(self.model, lora_config)
        
    def forward(self, input_ids, attention_mask, labels = None):
        """
        Forward del modelo. Devuelve loss si hay labels, o logits si no
        """
        return self.model(input_ids = input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
    def generate(self, input_text, max_length = 128):
        """
        Genera texto traducido dada una entradaa
        """
        self.model.eval()
        inputs = self.tokenizer(input_text, return_tensor='pt', padding= True)
        outputs = self.model.generate(**inputs, max_lenght=max_length)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
        
        return decoded