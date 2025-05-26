import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Dataset
import torchaudio
# from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim, Spectrogram, InverseSpectrogram
import librosa 
from src.sde import VE_SDE
from src.unet_NA import UNet
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import logging
import numpy as np
import soundfile as sf
import noisereduce as nr
logging.basicConfig(
    filename='test_log.txt',  
    level=logging.INFO,  
    format='%(asctime)s:%(levelname)s:%(message)s'
)
SAMPLE_RATE = 22050
class TIMITDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None,dataset_len=312):
        self.clean_dir = os.path.join(root_dir, subset, 'clean')
        self.noisy_dir = os.path.join(root_dir, subset, 'noisy')
        self.clean_files = sorted([os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) if f.endswith('.wav')])[:dataset_len]
        self.noisy_files = sorted([os.path.join(self.noisy_dir, f) for f in os.listdir(self.noisy_dir) if f.endswith('.wav')])[:dataset_len]
        print(f'Loaded {len(self.clean_files)} clean files and {len(self.noisy_files)} noisy files.')

        self.transform = transform

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_audio, sample_rate = librosa.load(self.clean_files[idx],sr=SAMPLE_RATE)
        noisy_audio, _ = librosa.load(self.noisy_files[idx],sr=SAMPLE_RATE)
        
        # clean_audio = clean_audio[:len(clean_audio)//3]
        # noisy_audio = noisy_audio[:len(noisy_audio)//3]
        
        clean_audio = librosa.feature.melspectrogram(y=clean_audio,n_mels=256, hop_length=507,sr=SAMPLE_RATE)
        noisy_audio = librosa.feature.melspectrogram(y=noisy_audio,n_mels=256, hop_length=507,sr=SAMPLE_RATE)
        
        if self.transform:
            clean_audio = self.transform(clean_audio)
            noisy_audio = self.transform(noisy_audio)
        # print(clean_audio.shape)
        # print(noisy_audio.shape)
        return torch.unsqueeze(torch.tensor(clean_audio),0), torch.unsqueeze(torch.tensor(noisy_audio),0)
def main(args):
    device = torch.device(f'cuda:{args.gpu}')
    print("Device:",device)
    ch = input("Continue?(y/n): ")
    if ch!='y':
        return
    
    c_in = 2
    time_dim = 256
    model = UNet(c_in=c_in, c_out=1, time_dim=time_dim,device=device).to(device)
    model.load_state_dict(torch.load(r'D:\GPU_Projects\Prism4\Speech\Speech-main\ckpts\New folder\ve_220.ckpt'))
    
    n_steps = 1400
    sde = sde = VE_SDE(theta=args.theta, rescale=True).to(device)
    test_data = TIMITDataset(root_dir=args.data_path, subset=args.subset,dataset_len=1)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True)
    # Create an "enhanced" directory if it doesn't exist
    os.makedirs('enhanced', exist_ok=True)
    os.makedirs(f'enhanced/a{n_steps}', exist_ok=True)
    folder = f'enhanced/a{n_steps}'
    if args.subset=='test':
        os.makedirs(f'enhanced/a{n_steps}/test', exist_ok=True)
        folder = f'enhanced/a{n_steps}/test'

    # PESQ Calculation
    pesq_scores = []
    pesq_scores1 = []
    # inverse_spec = InverseSpectrogram(n_fft=mel_transform.n_fft//2 + 1)

    with torch.no_grad():
        for idx, (clean, noisy) in enumerate(tqdm(test_dataloader)):
            x, y = clean.to(device), noisy.to(device)
            
            # Generate enhanced mel spectrogram using predictor_corrector_sample
            enhanced_mel = sde.predictor_corrector_sample(model, shape=y.shape, device=device, y=y, n_steps=n_steps)
            enhanced_mel = enhanced_mel.cpu().numpy()
            # print(enhanced_mel.squeeze(0).shape)
            # enhanced_mel = enhanced_mel.squeeze(0)
            # enhanced_mel = enhanced_mel.clamp(min=1e-8)
            # enhanced_mel = torch.nn.functional.softplus(enhanced_mel)
            # enhanced_mel = torch.sqrt(enhanced_mel)
            # print(enhanced_mel)
            # if torch.any(torch.isnan(enhanced_mel)) or torch.any(torch.isinf(enhanced_mel)):
            #     print("Enhanced mel contains NaNs or Infs")

            # Convert the mel spectrogram back to waveform
            enhanced_audio = librosa.feature.inverse.mel_to_audio(enhanced_mel.squeeze(), hop_length=507,sr=SAMPLE_RATE)
            y_in = y.cpu().numpy()
            y_in = librosa.feature.inverse.mel_to_audio(y_in.squeeze(), hop_length=507,sr=SAMPLE_RATE)
            
            x_in = x.cpu().numpy()
            x_in = librosa.feature.inverse.mel_to_audio(x_in.squeeze(), hop_length=507,sr=SAMPLE_RATE)
            # enhanced_audio = enhanced_audio.squeeze(0)
            enhanced_audio_path = os.path.join(folder, f'enhancedModel_Op_{idx}.wav')
            sf.write(enhanced_audio_path, enhanced_audio, SAMPLE_RATE)
            enhanced_audio = y_in-enhanced_audio
            # Save the enhanced audio file
            enhanced_audio_path = os.path.join(folder, f'enhancedN_{idx}.wav')
            x_audio_path = os.path.join(folder, f'clean_{idx}.wav')
            # enhanced_audio = torch.tensor(enhanced_audio)
            # torchaudio.save(enhanced_audio_path, enhanced_audio.cpu(), 16000)
            sf.write(enhanced_audio_path, enhanced_audio, SAMPLE_RATE)
            sf.write(x_audio_path, x_in, SAMPLE_RATE)
            # Calculate PESQ score
            pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
            # pesq_score = pesq(y.cpu().numpy(), enhanced_audio.cpu().numpy())
            print(f'Length of clean audio: {x_in.shape}')
            print(f'Length of enhanced audio: {enhanced_audio.shape}')

            
            #reloading
            enhanced_audio2, sample_rate = librosa.load(folder + f"/enhancedN_{idx}.wav",sr=16000)
            enhanced_audio3, sample_rate = librosa.load(folder + f"/enhancedModel_Op_{idx}.wav",sr=16000)
            x_in2, sample_rate = librosa.load(folder + f"/clean_{idx}.wav",sr=16000)
            pesq_score2 = pesq(torch.tensor(x_in2), torch.tensor(enhanced_audio3))
            enhanced_audio3 = nr.reduce_noise(y=enhanced_audio3,sr=sample_rate)
            enhanced_audio2 = nr.reduce_noise(y=enhanced_audio2,sr=sample_rate)
            enhanced_audio_path = os.path.join(folder, f'enhancedModel_Op_reduced{idx}.wav')
            sf.write(enhanced_audio_path, enhanced_audio3, SAMPLE_RATE)
            pesq_score = pesq(torch.tensor(x_in2), torch.tensor(enhanced_audio2))
            pesq_score1 = pesq(torch.tensor(x_in2), torch.tensor(enhanced_audio3))
            pesq_scores.append(pesq_score1)
            pesq_scores1.append(pesq_score2)
            logging.info(f'PESQ Score for Opsample {idx}: {pesq_score2}')
            
            print(f'PESQ Score for OPsampleReduced {idx}: {pesq_score1}')
            print(f'PESQ Score for sample_sub_Red {idx}: {pesq_score}')
            print(f'PESQ Score for OPsample {idx}: {pesq_score2}')
    # Log average PESQ score
    avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0
    avg_pesq1 = sum(pesq_scores1) / len(pesq_scores1) if pesq_scores else 0
    logging.info(f'Average PESQ Score: {avg_pesq1}')
    logging.info(f'Average PESQ Score_Reduced: {avg_pesq}')
    print(f'Average PESQ Score: {avg_pesq}')
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--theta', type=int, default=1.5)
    parser.add_argument('--subset', type=str, default='train')
    parser.add_argument('--data_path', type=str, required=True, help="Path to the root directory of TIMIT dataset")
    args = parser.parse_args()
    main(args)