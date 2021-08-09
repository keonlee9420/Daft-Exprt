import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    PhonemeEncoder,
    FrameDecoder,
    ProsodyEncoder,
    LocalProsodyPredictor,
    GaussianUpsampling,
)
from utils.tools import get_mask_from_lengths


class DaftExprt(nn.Module):
    """ Daft-Exprt """

    def __init__(self, preprocess_config, model_config):
        super(DaftExprt, self).__init__()
        self.model_config = model_config

        self.phoneme_encoder = PhonemeEncoder(model_config)
        self.prosody_encoder = ProsodyEncoder(preprocess_config, model_config)
        self.local_prosody_predictor = LocalProsodyPredictor(preprocess_config, model_config)
        self.gaussian_upsampling = GaussianUpsampling(preprocess_config, model_config)
        self.frame_decoder = FrameDecoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["prosody_encoder"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["prosody_encoder"]["encoder_hidden"],
                )
        self.regularization_weight = list()

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        ref_mels=None,
        ref_mel_lens=None,
        ref_max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        spker_embeds=None,
        ref_pitches=None,
        ref_energies=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        ref_mel_masks = get_mask_from_lengths(ref_mel_lens, ref_max_mel_len)

        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                spker_embeds = self.speaker_emb(speakers).unsqueeze(1)
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                spker_embeds = self.speaker_emb(spker_embeds).unsqueeze(1)

        gammas, betas, speaker_posterior = self.prosody_encoder(
            ref_mels, ref_max_mel_len, ref_mel_masks, ref_pitches, ref_energies, spker_embeds
        )

        output = self.phoneme_encoder(texts, src_masks, gammas, betas)

        p_predictions, e_predictions, log_d_predictions = self.local_prosody_predictor(
            output, src_masks, gammas, betas
        )

        output, attn, mel_lens, max_mel_len, mel_masks, d_rounded = self.gaussian_upsampling(
            output, p_predictions, e_predictions, log_d_predictions, p_targets, e_targets, d_targets,
            p_control, e_control, d_control, src_masks, ref_mel_lens, ref_max_mel_len, ref_mel_masks, src_lens
        )

        output, mel_masks = self.frame_decoder(output, mel_masks, gammas, betas)
        output = self.mel_linear(output)

        return (
            output,
            attn,
            speaker_posterior,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )