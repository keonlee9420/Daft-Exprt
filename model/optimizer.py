import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.l_anneal_steps = train_config["loss"]["anneal_steps"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = train_config["optimizer"]["init_lr"]
        self.anneal_lr = train_config["optimizer"]["anneal_lr"]

    def step_and_update_lr(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        if self.current_step > self.l_anneal_steps:
            lr = self.anneal_lr
            for s in self.anneal_steps:
                if self.current_step > s:
                    lr = lr * self.anneal_rate
        else:
            ratio = self.current_step / self.l_anneal_steps
            lr = self.init_lr + ratio * (self.anneal_lr - self.init_lr)
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr