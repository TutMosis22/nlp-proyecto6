import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.modelo_seq2seq import Seq2SeqModel

st.set_page_config(page_title="Traductor Comparativo", layout="centered")
st.title("Traductor Español-Inglés con Comparación de Modelos")

texto = st.text_area("Introduce un texto en Español o Inglés:", height=150)

if st.button("Traducir con ambos modelos"):
    with st.spinner("Cargando modelos..."):
        modelo_full = Seq2SeqModel(mode="full_finetune")
        modelo_lora = Seq2SeqModel(mode="lora")
    
    st.subheader("Resultados de Traducción")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fine-tuning completo**")
        resultado1 = modelo_full.generate([texto])[0]
        st.success(resultado1)
    
    with col2:
        st.markdown("**LoRA**")
        resultado2 = modelo_lora.generate([texto])[0]
        st.success(resultado2)
    
    st.subheader("Comparación")
    if resultado1.strip().lower() == resultado2.strip().lower():
        st.info("Ambos modelos generaron la misma traducción.")
    else:
        st.markdown("- Fine-tuning completo puede ser más preciso pero consume más recursos.")
        st.markdown("- LoRA es más liviano y rápido, con traducción **similar** en calidad.")