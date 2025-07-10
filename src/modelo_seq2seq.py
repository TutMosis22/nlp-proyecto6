import torch
from torch import nn
from transformers import T5ForConditionalGeneration, AutoTokenize

from peft import get_peft_model, LaraConfig, TaskType
#from transformers.adapters import AdapterConfig

import logging

logger = logging.getLogger(__name__)

class Seq2SeqModel(nn.Module):
    """
    Modelo Seq2Seq basado en T5-Small con tres modos de entrenaminto:
    - Fine-tuning
    - Adapters
    - LoRA
    """