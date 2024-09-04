import warnings

import json
import logging
import argparse
import torch


from lib.experiment import Experiment
from lib.config import Config
from lib.runner import Runner

# Ignore warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model GNN for predict num case HIV - C2DHIVFORECAST")
    parser.add_argument(
        "mode", choices=["train", "test", "plot"], help="Train or test")
    parser.add_argument(
        "--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument(
        "--root", type=str, default="./GNN/data", help="Root directory dataset")
    parser.add_argument(
        "--dataset", type=str, default="hcm_data_new", help="Dataset name")
    parser.add_argument(
        "--scale", action="store_true", default=False, help="Normalize data")
    parser.add_argument(
        "--ratio", type=float, default=0.8, help="Train test split")
    parser.add_argument(
        "--node_dim", type=int, default=128, help="Node embedding dimension")
    parser.add_argument(
        "--edge_dim", type=int, default=128, help="Edge embedding dimension")
    parser.add_argument(
        "--pooling", choices=["mean", "sag", "topk"], default="sag", help="pooling for graph")
    parser.add_argument(
        "--graph_block", choices=["transformer", "gat", "gat2", "gin"], default="transformer", help="Graph block al")
    parser.add_argument(
        "--val_on_epoch", type=int, default=5, help="Validation on epoch")
    parser.add_argument(
        "--train_epochs", type=int, default=1500, help="Number of training epochs")
    parser.add_argument(
        "--train_batch", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--test_batch", type=int, default=2, help="Testing batch size")
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--deterministic", action="store_true", default=False, help="Set random seed")
    parser.add_argument(
        "--ignore", choices=['infectious_object', 'occupation',
                             'infection_route', 'sex',
                             'phys_pos', 'age_grp', 'all', None],
        default=None, help="Ignore some features")
    parser.add_argument(
        "--offset", type=int, default=3, help="Offset for time series")
    parser.add_argument(
        "--offsetType", type=str, default="month", help="Offset type for time series")
    parser.add_argument(
        "--predictOffset", type=int, default=1, help="Predict offset for time series")

    args = parser.parse_args()
    with open("lib/configs/graph_block.json", "r") as f:
        graph_block_params = json.load(f)
        args.graph_block_params = graph_block_params[args.graph_block]
    with open("lib/configs/pooling.json", "r") as f:
        pooling_params = json.load(f)
        args.pooling_params = pooling_params[args.pooling]

    return args


def main():
    args = parse_args()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    exp = Experiment(args.exp_name, mode=args.mode,
                     model_checkpoint_interval=args.val_on_epoch)
    cfg = Config(args, device)
    runner = Runner(cfg, exp, device, epochs=args.train_epochs, val_on_epoch=args.val_on_epoch,
                    train_batch_size=args.train_batch, test_batch_size=args.test_batch, deterministic=args.deterministic, args=args)

    if args.mode == 'train':
        try:
            runner.train()
        except KeyboardInterrupt:
            logging.info('Training interrupted.')
    else:
        runner.test()
    return


if __name__ == "__main__":
    main()
