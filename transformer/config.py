from pathlib import Path
def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 20,
        "lr": 10**-4,
        "language": "vi",
        "src_len": 254,
        "tgt_len": 32,
        "d_model": 512,
        "train_data": "data/train_data.csv",
        "val_data": "data/val_data.csv",
        "model_folder": "data/test_newdata",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer/tokenizer_{0}.json",
        "chunking": True,
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        print("No weights files found in folder: ", model_folder)
        return None
    weights_files.sort()
    print(weights_files[-1])
    return str(weights_files[-1])
