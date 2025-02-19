from transformers import EncoderDecoderModel, AutoTokenizer
from underthesea import word_tokenize


def phobert2phobert_summarize(sentence):

    sentence = word_tokenize(sentence, format="text")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model = EncoderDecoderModel.from_pretrained(
        "phobert2phobert/weight_phobert2phobert/checkpoint-32950"
    )
    inputs = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str[0]
