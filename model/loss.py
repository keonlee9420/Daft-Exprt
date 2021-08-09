import torch
import torch.nn as nn
from torch.nn import functional as F


class DaftExprtLoss(nn.Module):
    """ Daft-Exprt Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(DaftExprtLoss, self).__init__()
        self.log_duration = preprocess_config["preprocessing"]["duration"]["log_duration"]
        # self.n_speaker = model_config["n_speaker"]
        self.anneal_steps = train_config["loss"]["anneal_steps"]
        self.lambda_f = train_config["loss"]["lambda_f"]
        self.lambda_a = train_config["loss"]["lambda_a"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.nll_loss = nn.NLLLoss()

    def anneal_lambda(self, step):
        lambda_f = self.lambda_f
        lambda_a = self.lambda_a if step > self.anneal_steps \
            else ((step / self.anneal_steps) * self.lambda_a)
        return lambda_f, lambda_a

    def forward(self, inputs, predictions, step, named_param):
        speaker_targets = inputs[2]
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            _,
            _,
        ) = inputs[6:]
        (
            mel_predictions,
            _,
            speaker_posteriors,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1) \
            if self.log_duration else duration_targets.float()
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        speaker_targets.requires_grad = False
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        pitch_predictions = pitch_predictions.masked_select(src_masks)
        pitch_targets = pitch_targets.masked_select(src_masks)
        energy_predictions = energy_predictions.masked_select(src_masks)
        energy_targets = energy_targets.masked_select(src_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mse_loss(mel_predictions, mel_targets) + self.mae_loss(mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        scale_reg = torch.sum(torch.square(named_param)) # L2-norm of the scaling parameters
        adv_loss = self.nll_loss(speaker_posteriors, speaker_targets)

        lambda_f, lambda_a = self.anneal_lambda(step)
        total_loss = (
            mel_loss + duration_loss + pitch_loss + energy_loss + lambda_f * scale_reg + lambda_a * adv_loss
        )

        return (
            total_loss,
            mel_loss,
            adv_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            lambda_f,
            lambda_a,
        )
