import argparse
import yaml

from src.baselines.bilstm_cnn.bilstm_cnn import BiLSTMCNNTrainer

def load_config(config_path):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def get_trainer(model_name, config):

    trainers = {
        "bilstm_cnn": BiLSTMCNNTrainer,
    }

    if model_name not in trainers:
        available = ", ".join(sorted(trainers))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    return trainers[model_name](config)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    trainer = get_trainer(
        config["model_name"],
        config
    )

    trainer.run()

if __name__ == "__main__":
    main()
