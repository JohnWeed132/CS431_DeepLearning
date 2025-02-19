from pathlib import Path
from tokenizers import Tokenizer
import torch
from transformer.model import build_transformer
from underthesea import word_tokenize


def transformer_summarize(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cpu")
    print("Using device:", device)
    sentence = word_tokenize(sentence, format="text")
    tokenizer = Tokenizer.from_file(
        str(Path("transformer/tokenizer/tokenizer_vi.json"))
    )
    print(str(Path("weight_transformer/tokenizer_vi.json")))
    model = build_transformer(
        tokenizer.get_vocab_size(),
        254,
        32,
        512,
    )
    # Load the pretrained weights
    model_filename = "transformer/weight_transformer/tmodel_20.pt"
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    print("ok1")
    model.load_state_dict(state["model_state_dict"])
    print("ok")
    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer.encode(sentence).ids
        if len(source) > 32 - 2:
            source = source[: 32 - 2]
        source = torch.cat(
            [
                torch.tensor([tokenizer.token_to_id("<s>")], dtype=torch.int64),
                torch.tensor(source, dtype=torch.int64),
                torch.tensor([tokenizer.token_to_id("</s>")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer.token_to_id("<pad>")] * (254 - len(source) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)
        source = source.unsqueeze(0)
        source_mask = (
            (source != tokenizer.token_to_id("<pad>"))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )
        print(source.shape)
        print(source_mask.shape)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = (
            torch.empty(1, 1)
            .fill_(tokenizer.token_to_id("</s>"))
            .type_as(source)
            .to(device)
        )

        # Print the source sentence and target start prompt

        # Generate the translation word by word
        while decoder_input.size(1) < 32:
            # build mask for target and calculate output
            decoder_mask = (
                (
                    torch.triu(
                        torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                        diagonal=1,
                    )
                    == 0
                )
                .type(torch.int)
                .type_as(source_mask)
                .to(device)
            )
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(device),
                ],
                dim=1,
            )

            # print the translated word
            print(f"{tokenizer.decode([next_word.item()])}", end=" ")

            # break if we predict the end of sentence token
            if next_word == tokenizer.token_to_id("</s>"):
                break

    # convert ids to tokens
    return tokenizer.decode(decoder_input[0].tolist())
