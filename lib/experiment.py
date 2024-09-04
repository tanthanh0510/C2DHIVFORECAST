import os
import re
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(self,
                 exp_name: str,
                 model_checkpoint_interval: int = 100,
                 mode: str = 'train',
                 exps_basedir: str = 'experiments',
                 tensorboard_dir: str = 'tensorboard') -> None:
        self.name = exp_name
        self.exp_dirpath = os.path.join(exps_basedir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, 'models')
        self.results_dirpath = os.path.join(self.exp_dirpath, 'results')
        self.log_path = os.path.join(self.exp_dirpath, f'log_{mode}.txt')
        self.tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_dir, exp_name))
        self.model_checkpoint_interval = model_checkpoint_interval
        self.setup_exp_dir()
        self.setup_logging()

    def setup_exp_dir(self) -> None:
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            os.makedirs(self.results_dirpath)

    def setup_logging(self) -> None:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[
                            file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

    def get_model_best_epoch(self):
        model_files = os.listdir(self.models_dirpath)
        model_files = [f for f in model_files if re.match(
            r'model_\d{4}.pt', f)]
        if not model_files:
            return 0
        model_files = sorted(model_files)
        best_model = model_files[-1]
        return int(best_model.split('.')[0].split('_')[1])

    def get_checkpoint_path(self, epoch: int):
        return os.path.join(self.models_dirpath, 'model_{:04d}.pt'.format(epoch))

    def get_epoch_model(self, epoch: int):
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        return torch.load(self.get_checkpoint_path(epoch), map_location=device)['model']

    def save_train_state(self,
                         epoch: int,
                         model: torch.nn.Module,
                         optimizer: torch.optim,
                         scheduler: torch.optim.lr_scheduler,
                         best_mse: float) -> None:
        train_state_path = self.get_checkpoint_path(epoch)
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mse': best_mse,
            }, train_state_path)

    def iter_end_callback(self,
                          epoch: int,
                          max_epochs: int,
                          iter_nb: int,
                          max_iter: int,
                          loss: float,
                          lr: float) -> None:
        line = 'Epoch [{}/{}] - Iter [{}/{}] - Loss: {:.5f} - Lr: {:.5f}'.format(
            epoch, max_epochs, iter_nb + 1, max_iter, loss, lr)
        self.logger.debug(line)
        overall_iter = (epoch * max_iter) + iter_nb
        self.tensorboard_writer.add_scalar(
            f'loss/total_loss', loss, overall_iter)
        self.tensorboard_writer.add_scalar(f'lr', lr, overall_iter)

    def epoch_start_callback(self, epoch: int, max_epochs: int) -> None:
        self.logger.debug('Epoch [%d/%d] starting.', epoch, max_epochs)

    def epoch_end_callback(self,
                           epoch: int,
                           max_epochs: int,
                           model: torch.nn.Module,
                           optimizer: torch.optim,
                           scheduler: torch.optim.lr_scheduler,
                           best_mse: float) -> None:
        self.logger.debug('Epoch [%d/%d] finished.\n', epoch, max_epochs)

        self.save_train_state(epoch, model, optimizer,
                              scheduler, best_mse)

    def delete_model(self,
                     is_not_delete_model: bool,
                     epoch: int) -> None:
        current_model = 'model_{:04d}.pt'.format(epoch)

        if len(os.listdir(self.models_dirpath)):
            for content in os.listdir(self.models_dirpath):
                if (content != current_model and is_not_delete_model) or (content == current_model and not is_not_delete_model):
                    os.remove(os.path.join(self.models_dirpath, content))

    def train_start_callback(self) -> None:
        self.logger.debug('Beginning training session.\n')

    def train_end_callback(self) -> None:
        self.logger.debug('Training session finished.\n')

    def eval_start_callback(self) -> None:
        self.logger.debug('Beginning testing session.\n')

    def eval_end_callback(self,
                          mse_score: float,
                          rmse_score: float,
                          r2_score: float,
                          epoch_evaluated: int,
                          mode: str = "validation") -> None:

        # log tensorboard metrics
        self.tensorboard_writer.add_scalar(
            f'MSE_{mode}', mse_score, epoch_evaluated)
        self.tensorboard_writer.add_scalar(
            f'RMSE_{mode}', rmse_score, epoch_evaluated)
        self.tensorboard_writer.add_scalar(
            f'r2_score_{mode}', r2_score, epoch_evaluated)

        self.logger.debug(
            f'{mode} session finished on model after epoch {epoch_evaluated}.\n')

        self.logger.info(
            f"MSE {mode}: {mse_score:.5f}, RMSE {mode}:{rmse_score:.5f}, R2 {mode}:{r2_score:.5f}\n")
