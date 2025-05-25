# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os

import torch
import torch.utils.data
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import tacotron2_common.layers as layers
from tacotron2_common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from tacotron2.text import text_to_sequence

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, args):
        self.args = args
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        len_text = len(text)
        text = self.get_text(text)
        mel, audio_segment = self.get_mel_audio(audiopath)
        emotion = int(os.path.basename(audiopath).split('.')[0].split('_')[-1])
        return (text, mel, len_text, emotion, audio_segment)
    
    def get_mel_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        
        # Take segment
        if audio.size(0) >= self.args.segment_length:
            max_audio_start = audio.size(0) - self.args.segment_length
            audio_start = torch.randint(0, max_audio_start + 1, size=(1,)).item()
            audio_segment = audio[audio_start:audio_start+self.args.segment_length]
        else:
            audio_segment = torch.nn.functional.pad(
                audio, (0, self.args.segment_length - audio.size(0)), 'constant').data
        audio_segment = audio_segment / self.max_wav_value

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        
        return melspec, audio_segment

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        
        # fig = plt.Figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # ax.set_title(filename)
        # p = librosa.display.specshow(librosa.amplitude_to_db(melspec, ref=np.max), ax=ax, y_axis='log', x_axis='time')
        # fig.savefig('/home/madusov/vkr/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/mel.png')
        # print('AAAAa')
        return melspec
    
    # def get_mel(self, fpath):
    #     '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    #     Args:
    #     sound_file: A string. The full path of a sound file.
    #     Returns:
    #     mel: A 2d array of shape (T, n_mels) <- Transposed
    #     mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    #     '''

    #     # Loading sound file
    #     y, sr = librosa.load(fpath, sr=self.args.sampling_rate)

    #     # Trimming
    #     y, _ = librosa.effects.trim(y)

    #     # Preemphasis
    #     y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    #     # stft
    #     linear = librosa.stft(y=y,
    #                         n_fft=self.args.filter_length,
    #                         hop_length=self.args.hop_length,
    #                         win_length=self.args.win_length)

    #     # magnitude spectrogram
    #     mag = np.abs(linear)  # (1+n_fft//2, T)

    #     # mel spectrogram
    #     mel_basis = librosa.filters.mel(sr=self.args.sampling_rate, 
    #                                     n_fft=self.args.filter_length, 
    #                                     n_mels=80)  # (n_mels, 1+n_fft//2)
    #     mel = np.dot(mel_basis, mag)  # (n_mels, t)

    #     # to decibel
    #     mel = 20 * np.log10(np.maximum(1e-5, mel))
    #     mag = 20 * np.log10(np.maximum(1e-5, mag))

    #     # normalize
    #     mel = np.clip((mel - 20 + 100) / 100, 1e-8, 1)
    #     mag = np.clip((mag - 20 + 100) / 100, 1e-8, 1)

    #     # Transpose
    #     mel = mel.T.astype(np.float32)  # (T, n_mels)
    #     mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)


    #     fig = plt.Figure()
    #     canvas = FigureCanvas(fig)
    #     ax = fig.add_subplot(111)
    #     ax.set_title(fpath)
    #     p = librosa.display.specshow(librosa.amplitude_to_db(torch.tensor(mel).permute(1,0), ref=np.max), ax=ax, y_axis='log', x_axis='time')
    #     fig.savefig('/home/madusov/vkr/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/mel.png')
    #     print('AAAAAAAAAAAAAAA')
        
    #     # print(torch.tensor(mel).permute(1,0).shape)
    #     return torch.tensor(mel).permute(1,0)

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        # collate emotions
        # emotions = [x[3] for x in batch]
        # emotions = torch.Tensor(emotions)

        # audio_segments = [x[4] for x in batch]
        # audio_segments = torch.stack(audio_segments, dim=0)

        emotions = []
        audio_segments = []
        for idx in ids_sorted_decreasing:
            emotions.append(batch[idx][3])
            audio_segments.append(batch[idx][4])
        emotions = torch.Tensor(emotions)
        audio_segments = torch.stack(audio_segments, dim=0)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x, emotions, audio_segments

def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x, emotions, audio_segments = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    len_x = torch.sum(output_lengths)
    emotions = to_gpu(emotions).long()
    audio_segments = to_gpu(audio_segments)

    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths, audio_segments)
    y = (mel_padded, gate_padded)
    return (x, y, len_x, emotions)
