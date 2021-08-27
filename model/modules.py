import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths

from .blocks import (
    GradientReversalLayer,
    FiLM,
    LinearNorm,
    ConvNorm,
    FFTBlock,
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class SpeakerClassifier(nn.Module):
    """ Speaker Classifier """

    def __init__(self, model_config):
        super(SpeakerClassifier, self).__init__()
        n_speaker = model_config["n_speaker"]
        input_dim = model_config["prosody_encoder"]["encoder_hidden"]
        self.hidden = model_config["prosody_encoder"]["encoder_hidden"]

        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_dim, self.hidden)),
            ('ln1', nn.LayerNorm(self.hidden)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.hidden, self.hidden)),
            ('ln2', nn.LayerNorm(self.hidden)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.hidden, n_speaker)),
            ('softmax', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        # GRL
        rev_x = self.grl(x)
        # Calculate augmentation posterior
        score = self.classifier(rev_x)
        if len(score.size()) > 2:
            score = score.mean(dim=1)
        return score # [batch, 2]


class PhonemeEncoder(nn.Module):
    """ Phoneme Encoder """

    def __init__(self, config):
        super(PhonemeEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, gammas, betas, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, gammas, betas, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class FrameDecoder(nn.Module):
    """ Frame Decoder """

    def __init__(self, config):
        super(FrameDecoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, gammas, betas, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, gammas, betas, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask


class ProsodyEncoder(nn.Module):
    """ Prosody Encoder """

    def __init__(self, preprocess_config, model_config):
        super(ProsodyEncoder, self).__init__()

        self.max_seq_len = model_config["max_seq_len"] + 1
        n_conv_layers = model_config["prosody_encoder"]["conv_layer"]
        kernel_size = model_config["prosody_encoder"]["conv_kernel_size"]
        n_layers = model_config["prosody_encoder"]["encoder_layer"]
        n_head = model_config["prosody_encoder"]["encoder_head"]
        self.d_model = model_config["prosody_encoder"]["encoder_hidden"]
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_k = d_v = (self.d_model // n_head)
        self.filter_size = model_config["prosody_encoder"]["conv_filter_size"]
        dropout = model_config["prosody_encoder"]["dropout"]

        self.speaker_classifier = SpeakerClassifier(model_config)

        self.p_embedding = ConvNorm(
            1,
            self.filter_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.e_embedding = ConvNorm(
            1,
            self.filter_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.layer_stack = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNorm(
                            n_mel_channels if i == 0 else self.filter_size,
                            self.filter_size,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                            dilation=1,
                            transform=True,
                        ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(dropout),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(self.max_seq_len, self.filter_size).unsqueeze(0),
            requires_grad=False,
        )

        self.fftb_linear = LinearNorm(self.filter_size, self.d_model)
        self.fftb_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model, n_head, d_k, d_v, self.filter_size, [kernel_size, kernel_size], dropout=dropout, film=False
                )
                for _ in range(n_layers)
            ]
        )

        self.feature_wise_affine = LinearNorm(self.d_model, 2 * self.d_model)

    def forward(self, mel, max_len, mask, pitch, energy, spker_embed):

        batch_size = mel.shape[0]
        pitch = self.p_embedding(pitch.unsqueeze(-1))
        energy = self.e_embedding(energy.unsqueeze(-1))

        # -- Prepare Input
        enc_seq = mel
        for enc_layer in self.layer_stack:
            enc_seq = enc_layer(enc_seq)
        enc_seq = enc_seq + pitch + energy

        if mask is not None:
            enc_seq = enc_seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_seq = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.filter_size
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_seq = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        enc_seq = self.fftb_linear(enc_seq)
        for fftb_layer in self.fftb_stack:
            enc_seq, _ = fftb_layer(
                enc_seq, mask=mask, slf_attn_mask=slf_attn_mask
            )

        # -- Avg Pooling
        prosody_vector = enc_seq.mean(dim=1, keepdim=True) # [B, 1, H]

        # -- Speaker Classifier
        speaker_posterior = self.speaker_classifier(prosody_vector.squeeze(1))

        # -- Feature-wise Affine
        gammas, betas = torch.split(
            self.feature_wise_affine(prosody_vector + spker_embed), self.d_model, dim=-1
        )

        return gammas, betas, speaker_posterior


class LocalProsodyPredictor(nn.Module):
    """ Local Prosody Predictor """

    def __init__(self, preprocess_config, model_config):
        super(LocalProsodyPredictor, self).__init__()
        self.prosody_predictor = ProsodyPredictor(model_config)
        # self.p_predictor = ProsodyPredictor(model_config)
        # self.e_predictor = ProsodyPredictor(model_config)
        # self.d_predictor = ProsodyPredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level == "phoneme_level"
        assert self.energy_feature_level == "phoneme_level"

    def forward(
        self,
        x,
        src_mask,
        gammas,
        betas,
    ):

        pitch_prediction, energy_prediction, log_duration_prediction = \
            [pred.squeeze(-1) for pred in torch.split(self.prosody_predictor(x, src_mask, gammas, betas), 1, dim=-1)]

        # pitch_prediction = self.p_predictor(x, src_mask, gammas, betas).squeeze(-1)
        # energy_prediction = self.e_predictor(x, src_mask, gammas, betas).squeeze(-1)
        # log_duration_prediction = self.d_predictor(x, src_mask, gammas, betas).squeeze(-1)

        return (
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
        )


class ProsodyPredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(ProsodyPredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["prosody_predictor"]["filter_size"]
        self.kernel = model_config["prosody_predictor"]["kernel_size"]
        self.dropout = model_config["prosody_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvNorm(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            transform=True,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        ConvNorm(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            transform=True,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )
        self.film = FiLM()
        self.linear_layer = LinearNorm(self.filter_size, 3)
        # self.linear_layer = LinearNorm(self.filter_size, 1)

    def forward(self, encoder_output, mask, gammas, betas):
        out = self.conv_layer(encoder_output)
        out = self.film(out, gammas, betas)
        out = self.linear_layer(out)

        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)

        return out


class GaussianUpsampling(nn.Module):
    """ Gaussian Upsampling """

    def __init__(self, preprocess_config, model_config):
        super(GaussianUpsampling, self).__init__()
        kernel_size = model_config["prosody_predictor"]["kernel_size"]
        self.log_duration = preprocess_config["preprocessing"]["duration"]["log_duration"]
        self.prosody_hidden = model_config["transformer"]["encoder_hidden"]

        self.p_embedding = ConvNorm(
            1,
            self.prosody_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.e_embedding = ConvNorm(
            1,
            self.prosody_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.d_embedding = ConvNorm(
            1,
            self.prosody_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.range_param_predictor_paper = nn.Sequential(
            nn.Linear(self.prosody_hidden, 1),
            nn.Softplus(),
        )
        self.range_param_predictor = RangeParameterPredictor(model_config)

    def get_alignment_energies(self, gaussian, frames):
        log_prob = gaussian.log_prob(frames)
        energies = log_prob.exp()  # [B, L, T]
        return energies

    def forward(self,
        encoder_outputs,
        p_prediction,
        e_prediction,
        d_prediction,
        p_targets=None,
        e_targets=None,
        d_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        src_mask=None,
        mel_len=None,
        max_mel_len=None,
        mel_mask=None,
        src_len=None,
    ):
        device = p_prediction.device

        if d_target is not None:
            p_prediction = p_targets
            e_prediction = e_targets
            d_prediction = d_target.float()
        else:
            p_prediction = p_prediction * p_control
            e_prediction = e_prediction * e_control
            d_prediction = torch.clamp(
                (torch.round(torch.exp(d_prediction) - 1) * d_control),
                min=0
            ).float() if self.log_duration else (d_prediction * d_control)
            mel_len = d_prediction.sum(-1).int().to(device)
            max_mel_len = mel_len.max().item()
            mel_mask = get_mask_from_lengths(mel_len, max_mel_len)

        # Prosody Projection
        p_embed = self.p_embedding(p_prediction.unsqueeze(-1))
        e_embed = self.e_embedding(e_prediction.unsqueeze(-1))
        d_embed = self.d_embedding(d_prediction.unsqueeze(-1))

        # Range Prediction
        s_input = p_embed + e_embed + d_embed + encoder_outputs
        s = self.range_param_predictor(s_input, src_len, src_mask).unsqueeze(-1) if src_len is not None \
            else self.range_param_predictor_paper(s_input)
        if src_mask is not None:
            s = s.masked_fill(src_mask.unsqueeze(-1), 1e-8)

        # Gaussian Upsampling
        t = torch.sum(d_prediction, dim=-1, keepdim=True) #[B, 1]
        e = torch.cumsum(d_prediction, dim=-1).float() #[B, L]
        c = e - 0.5 * d_prediction #[B, L]
        t = torch.arange(1, torch.max(t).item()+1, device=device) # (1, ..., T)
        t = t.unsqueeze(0).unsqueeze(1) #[1, 1, T]
        c = c.unsqueeze(2)

        g = torch.distributions.normal.Normal(loc=c, scale=s)

        w = self.get_alignment_energies(g, t)  # [B, L, T]

        if src_mask is not None:
            w = w.masked_fill(src_mask.unsqueeze(-1), 0.0)

        attn = w / (torch.sum(w, dim=1).unsqueeze(1) + 1e-8)  # [B, L, T]
        out = torch.bmm(attn.transpose(1, 2), p_embed + e_embed + encoder_outputs)

        return out, attn, mel_len, max_mel_len, mel_mask, d_prediction


class RangeParameterPredictor(nn.Module):
    """ Range Parameter Predictor """

    def __init__(self, model_config):
        super(RangeParameterPredictor, self).__init__()
        encoder_hidden = model_config["transformer"]["encoder_hidden"]
        prosody_hidden = model_config["transformer"]["encoder_hidden"]

        self.range_param_lstm = nn.LSTM(
            encoder_hidden,
            int(prosody_hidden / 2), 2,
            batch_first=True, bidirectional=True
        )
        self.range_param_proj = nn.Sequential(
            LinearNorm(prosody_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, encoder_output, output_len, mask):

        range_param_input = encoder_output

        if self.training:
            output_len = output_len.cpu().numpy()
            range_param_input = nn.utils.rnn.pack_padded_sequence(
                range_param_input, output_len, batch_first=True)

        self.range_param_lstm.flatten_parameters()
        range_param_prediction, _ = self.range_param_lstm(range_param_input)  # [B, L, channels]
        if self.training:
            range_param_prediction, _ = nn.utils.rnn.pad_packed_sequence(
                range_param_prediction, batch_first=True)

        range_param_prediction = self.range_param_proj(range_param_prediction)  # [B, L, 1]
        range_param_prediction = range_param_prediction.squeeze(-1)  # [B, L]
        if mask is not None:
            range_param_prediction = range_param_prediction.masked_fill(mask, 1e-8)

        return range_param_prediction
