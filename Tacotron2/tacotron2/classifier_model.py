from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
import sys
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/../'))
from tacotron2.GST import GST
from tacotron2.emotion_classifier import EmotionClassifier
from tacotron2.vqvae.vqvae import VQVAE
from tacotron2.m5.m5 import M5


class GSTClassifier(nn.Module):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 attention_rnn_dim, attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 decoder_rnn_dim, prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions, decoder_no_early_stopping,
                 ref_enc_filters, n_mels, token_num, num_heads):
        super(GSTClassifier, self).__init__()
        
        print(f'{encoder_embedding_dim=}')
        print(f'{ref_enc_filters=}')
        print(f'{token_num=}')
        print(f'{num_heads=}')
        # self.gst = GST(ref_enc_filters=ref_enc_filters, n_mels=n_mels, emb_size=encoder_embedding_dim, token_num=token_num, num_heads=num_heads)
        self.vqvae = VQVAE(h_dim=128,
                           res_h_dim=32,
                           n_res_layers=2,
                           n_embeddings=512,
                           embedding_dim=512,
                           beta=0.25)
        # self.m5 = M5()
        self.emotion_classifier = EmotionClassifier(512)
        # self.emotion_classifier = EmotionClassifier(64)

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, output_lengths, audio_segments = inputs

        # style_embed = self.vqvae(inputs.unsqueeze(1))
        style_embed = self.vqvae(targets.permute(0,2,1).unsqueeze(1))
        # print(style_embed.sum(dim=2).sum(dim=2).shape)
        # style_embed = self.gst(targets.permute(0,2,1))  # [N, 256]
        # emotion_logits = self.emotion_classifier(style_embed.sum(dim=2))

        emotion_logits = self.emotion_classifier(style_embed.sum(dim=2).sum(dim=2))

        # m5
        # style_embed = self.m5(audio_segments.unsqueeze(1))
        # emotion_logits = self.emotion_classifier(style_embed)

        return emotion_logits
