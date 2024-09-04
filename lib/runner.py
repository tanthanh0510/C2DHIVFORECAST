import random
import numpy as np
from tqdm import tqdm
import logging
import json
import torch
from torch.utils.data import DataLoader

from utils.evaluation_metrics import mse, rmse, r2
from utils.create_data import collate

from lib.config import Config
from lib.experiment import Experiment

SEED = 42


class Runner:
    def __init__(self,
                 cfg: None or Config = None,
                 exp: None or Experiment = None,
                 device: None or torch.device = None,
                 epochs: int = 100,
                 val_on_epoch: int = 5,
                 train_batch_size: int = 4,
                 test_batch_size: int = 4,
                 deterministic: bool = False,
                 args: dict = None) -> None:
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_on_epoch = val_on_epoch
        self.logger = logging.getLogger(__name__)
        self.args = args

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def test(self) -> str:

        data_loader = self.get_full_dataloader()
        model = self.cfg.get_model().to(self.device)
        best_epoch = self.exp.get_model_best_epoch()
        model.load_state_dict(
            self.exp.get_epoch_model(best_epoch), strict=False)
        ground_truth = torch.Tensor().to(self.device)
        prediction = torch.Tensor().to(self.device)
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                data.to(self.device)
                output = model(data)
                print(output.size())
                print(data.y.size())
                output = output.view(23, -1)  # (23, batch_size)
                prediction = torch.cat((prediction, output), 1)
                ground_truth = torch.cat(
                    (ground_truth, data.y.view(23, -1)), 1)
                print(f"Prediction: {prediction.size()}")
                print(f"Ground truth: {ground_truth.size()}")
        prediction = prediction.cpu()
        ground_truth = ground_truth.cpu()
        mse_score, rmse_score, r2_score = mse(ground_truth.numpy().flatten(), prediction.numpy().flatten()), rmse(
            ground_truth.numpy().flatten(), prediction.numpy().flatten()), r2(ground_truth.numpy().flatten(), prediction.numpy().flatten())
        self.logger.info(
            f"Performance on test set: MSE: {mse_score:.5f}, RMSE: {rmse_score:.5f}, R2: {r2_score:.5f}.\n")
        # save matrix to npy file
        np.save(f"{self.exp.exp_dirpath}/prediction.npy", prediction.numpy())
        np.save(f"{self.exp.exp_dirpath}/ground_truth.npy",
                ground_truth.numpy())
        return

    def train(self) -> tuple:
        best_epoch = 0
        starting_epoch = 1
        max_epochs = self.epochs

        self.exp.train_start_callback()
        train_loader = self.get_train_dataloader()

        model = self.cfg.get_model().to(self.device)
        print(model)
        loss_fn = self.cfg.get_loss_function()
        total_params = model.parameters()
        optimizer = self.cfg.get_optimizer(total_params)
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        best_mse = float("inf")
        best_rmse = float("inf")
        best_r2 = float("-inf")

        for epoch in range(starting_epoch, max_epochs + 1, 1):
            model.train()
            self.exp.epoch_start_callback(epoch, max_epochs)
            pbar = tqdm(train_loader, desc=f"Train on epoch {epoch}")
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                data.to(self.device)
                output = model(data, feature_ignore=self.args.ignore)
                loss = loss_fn(
                    output, data.y.view(-1, 1).float().to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(loss=loss.item())

                self.exp.iter_end_callback(epoch, max_epochs, i, len(
                    train_loader), loss.item(), optimizer.param_groups[0]["lr"])

            # Validation each interval epoch
            self.exp.epoch_end_callback(
                epoch, max_epochs, model, optimizer, scheduler, best_mse)

            mse_score_val, rmse_score_val, r2_val = self.eval(epoch=epoch)

            # Save model if achieve the best result on MSE
            if mse_score_val <= best_mse:
                best_mse = mse_score_val
                best_rmse = rmse_score_val
                best_r2 = r2_val
                best_epoch = epoch
            self.exp.delete_model(mse_score_val == best_mse, epoch)

        self.exp.train_end_callback()
        self.logger.info(
            f"Best performance on validation at epoch {best_epoch} th \n")
        self.logger.info(
            f"Best performance on validation set: MSE: {best_mse:.5f}, RMSE: {best_rmse:.5f}, R2: {best_r2:.5f}.\n")
        # write best performance to json file
        with open(f"{self.exp.exp_dir}/best_performance.json", "w") as f:
            json.dump({"mse": best_mse, "rmse": best_rmse, "r2": best_r2}, f)
        return best_mse, best_rmse, best_r2

    def eval(self,
             epoch: int or None = None) -> tuple:
        dataloader = self.get_val_dataloader()
        model = self.cfg.get_model()
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model.to(self.device)

        des = f"Validation phrase on epoch {epoch}"

        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        model.eval()
        self.exp.eval_start_callback()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=des)
            for data in pbar:
                data.to(self.device)
                output = model(data, feature_ignore=self.args.ignore)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat(
                    (total_labels, data.y.view(-1, 1).cpu()), 0)

        ground_truth = total_labels.numpy().flatten()
        prediction = total_preds.numpy().flatten()
        mse_score, rmse_score, r2_score = mse(ground_truth, prediction), rmse(
            ground_truth, prediction), r2(ground_truth, prediction)
        self.exp.eval_end_callback(
            mse_score, rmse_score, r2_score, epoch)
        return mse_score, rmse_score, r2_score

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.cfg.get_dataset(mode='train')

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_batch_size,
                                  shuffle=True,
                                  collate_fn=collate,
                                  num_workers=8,
                                  worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_val_dataloader(self) -> DataLoader:
        val_dataset = self.cfg.get_dataset(mode='test')

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_batch_size,
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=8,
                                worker_init_fn=self._worker_init_fn_)
        return val_loader

    def get_full_dataloader(self) -> DataLoader:
        full_dataset = self.cfg.get_dataset(mode='full')

        full_loader = DataLoader(dataset=full_dataset,
                                 batch_size=self.test_batch_size,
                                 shuffle=False,
                                 collate_fn=collate,
                                 num_workers=8,
                                 worker_init_fn=self._worker_init_fn_)
        return full_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
