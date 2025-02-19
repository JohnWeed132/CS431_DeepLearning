import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SummarizeDataset(Dataset):

    def __init__(self, ds, tokenizer, src_len, tgt_len, chunking=False):
        super().__init__()
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.chunking = chunking
        self.ds = ds
        self.tokenizer = tokenizer

        self.sos_token = torch.tensor([tokenizer.token_to_id("<s>")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer.token_to_id("</s>")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer.token_to_id("<pad>")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["original"]
        tgt_text = src_target_pair["summary"]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids
        if self.chunking:
            if len(enc_input_tokens) > self.src_len - 2:
                enc_input_tokens = enc_input_tokens[: self.src_len - 2]
            if len(dec_input_tokens) > self.tgt_len - 1:
                dec_input_tokens = dec_input_tokens[: self.tgt_len - 1]
        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = (
            self.src_len - len(enc_input_tokens) - 2
        )  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.tgt_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.src_len
        assert decoder_input.size(0) == self.tgt_len
        assert label.size(0) == self.tgt_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
