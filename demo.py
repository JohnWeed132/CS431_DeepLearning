import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformer.transformer_summarize import transformer_summarize
from phobert2phobert.phobert2phobert_summarize import phobert2phobert_summarize
import time

# Streamlit App Layout
st.title("Text Summarization Demo")

# Input văn bản từ người dùng
txt = st.text_area("Nhập văn bản cần tóm tắt:", height=200)

# Chọn mô hình tóm tắt
model_choice = st.selectbox(
    "Chọn mô hình tóm tắt", ["All", "Phobert2Phobert", "Transformers"]
)

# Khi nhấn nút "Tóm tắt"
if st.button("Tóm tắt"):
    if not txt:
        st.warning("Vui lòng nhập văn bản cần tóm tắt!")
    else:
        st.subheader("Tóm tắt:")
        with st.spinner("Đang xử lý..."):
            if model_choice == "Transformers" or model_choice == "All":
                summary_trans = transformer_summarize(txt)
                st.write(f"Transformers: {summary_trans}")
            if model_choice == "Phobert2Phobert" or model_choice == "All":
                summary_pho = phobert2phobert_summarize(txt)
                st.write(f"Phobert2Phobert: {summary_pho}")
