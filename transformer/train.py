from model import build_transformer
from dataset import SummarizeDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd

# Huggingface datasets and tokenizers
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("<s>")
    eos_idx = tokenizer_tgt.token_to_id("</s>")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg("-" * console_width)
            print_msg(f"{encoder_input.shape}")
            print_msg(f"{encoder_mask.shape}")
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_all_sentences(ds):
    for item in ds:
        yield item["original"] + item["summary"]


def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config["tokenizer_file"].format(config["language"]))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print("Success load tokenizer")
    return tokenizer


def get_ds(config):
    # Load the dataset
    train_ds_raw = pd.read_csv(config["train_data"])
    val_ds_raw = pd.read_csv(config["val_data"])
    ds_raw = pd.concat([train_ds_raw, val_ds_raw], axis=0, ignore_index=True)

    train_ds_raw = Dataset.from_pandas(train_ds_raw)
    val_ds_raw = Dataset.from_pandas(val_ds_raw)
    ds_raw = Dataset.from_pandas(ds_raw)
    # Build tokenizers
    tokenizer = get_or_build_tokenizer(config, ds_raw)

    train_ds = SummarizeDataset(
        train_ds_raw,
        tokenizer,
        config["src_len"],
        config["tgt_len"],
        config["chunking"],
    )
    val_ds = SummarizeDataset(
        val_ds_raw, tokenizer, config["src_len"], config["tgt_len"], config["chunking"]
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    i_src = -1
    i_tgt = -1
    for i, item in enumerate(train_ds_raw):
        src_ids = tokenizer.encode(item["original"]).ids
        tgt_ids = tokenizer.encode(item["summary"]).ids
        if max_len_src < len(src_ids):
            max_len_src = len(src_ids)
            i_src = i
        if max_len_tgt < len(tgt_ids):
            max_len_tgt = len(tgt_ids)
            i_tgt = i
        # max_len_src = max(max_len_src, len(src_ids))
        # max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src} at index {i_src}")
    print(f"Max length of target sentence: {max_len_tgt} at index {i_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_src_len):
    model = build_transformer(
        vocab_src_len,
        config["src_len"],
        config["tgt_len"],
        d_model=config["d_model"],
    )
    return model


def train_model(config):
    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        print("ok")
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id("<pad>"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch["label"].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model,
            val_dataloader,
            tokenizer,
            config["tgt_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    config["batch_size"] = 1
    config["preload"] = "latest"
    config["num_epochs"] = 3
    config["chunking"] = True
    config["model_folder"] = "test_newdata"
    config["src_len"] = 254
    config["tgt_len"] = 32
    train_model(config)
    val = pd.read_csv("val_data.csv")
    val = val.head(100)
    pd.DataFrame.to_csv(val, "val_data.csv", index=False)
