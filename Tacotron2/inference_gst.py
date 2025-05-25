from tacotron2.test_model import GSTClassifier
import argparse
import models


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--fp16', action='store_true',
                        help='Run inference with mixed precision')
    run_mode.add_argument('--cpu', action='store_true',
                        help='Run inference on CPU')

    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--ref-path', type=str, default=None)

    return parser


class MelLoader():
    def __init__(self, text_cleaners, max_wav_value, sampling_rate, filter_length, hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax):
        self.text_cleaners = text_cleaners
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.stft = layers.TacotronSTFT(
            filter_length, hop_length, win_length,
            n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec


def main_gst():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()


    model_parser = models.model_parser('Tacotron2', parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config('Tacotron2', model_args)

    
    model = GSTClassifier(**model_config)
    model.load_state_dict('/home/madusov/vkr/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/output/checkpoint_Tacotron2_70.pt')
    model.eval()

    with open('config.json') as f:
        audio_config = json.load(f)
    
    loader = MelLoader(text_cleaners=['english_cleaners'], 
                       max_wav_value=audio_config['audio']['max-wav-value'], 
                       sampling_rate=audio_config['audio']['sampling-rate'], 
                       filter_length=audio_config['audio']['filter-length'], 
                       hop_length=audio_config['audio']['hop-length'], 
                       win_length=audio_config['audio']['win-length'], 
                       n_mel_channels=80, 
                       mel_fmin=audio_config['audio']['mel-fmin'], 
                       mel_fmax=audio_config['audio']['mel-fmax'])
    ref_mel = loader.get_mel(args.ref_path)
    ref_mel = ref_mel.unsqueeze(0)

    if not args.cpu:
        ref_mel = ref_mel.to('cuda')


    with torch.no_grad():
        model()

   


if __name__ == '__main__':
    main_gst()
