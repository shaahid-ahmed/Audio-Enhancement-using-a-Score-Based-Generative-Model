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
import logging
import numpy as np
import soundfile as sf
# Configure logging
logging.basicConfig(
    filename='training_log_NA.txt',  
    level=logging.INFO,  
    format='%(asctime)s:%(levelname)s:%(message)s'
)
SAMPLE_RATE = 16000
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
        clean_audio = librosa.feature.melspectrogram(y=clean_audio,n_mels=256, hop_length=521,sr=sample_rate)
        noisy_audio = librosa.feature.melspectrogram(y=noisy_audio,n_mels=256, hop_length=521,sr=sample_rate)
        if self.transform:
            clean_audio = self.transform(clean_audio)
            noisy_audio = self.transform(noisy_audio)
        # print(clean_audio.shape)
        # print(noisy_audio.shape)
        return torch.unsqueeze(torch.tensor(clean_audio),0), torch.unsqueeze(torch.tensor(noisy_audio),0)

# def preprocess_audio(y,n_fft=126, hop_length=3230):
#     # Apply Spectrogram transformation
#     return librosa.stft(y,n_fft=n_fft, hop_length=hop_length)

def main(args):
    os.makedirs('ckpts', exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)
    # # Define the MelSpectrogram transformation
    # mel_transform = preprocess_audio()

    # Load dataset
    train_data = TIMITDataset(root_dir=args.data_path, subset='train',dataset_len=3000)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    test_data = TIMITDataset(root_dir=args.data_path, subset='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True)
    # Define the UNet model for mel spectrogram input dimensions
    c_in = 2
    time_dim = 256
    # model = UNet(c_in=c_in, c_out=1, time_dim=time_dim,device=device).to(device)
    model = UNet(c_in=c_in, c_out=1, time_dim=time_dim,device=device).to(device)
    model.load_state_dict(torch.load(r'D:\GPU_Projects\Prism4\Speech\Speech-main\ckpts\ve_80.ckpt'))
    for param in model.parameters():
        param.requires_grad = True
    # Select the SDE
    if args.sde == 've':
        sde = VE_SDE(theta=args.theta, rescale=True).to(device)
    else:
        raise ValueError('Invalid SDE type')

    # Optimizer and EMA
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-12)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    criterion = nn.MSELoss()
    t_eps = 0.03

    # Training loop
    for epoch in range(200):
        logging.info(f'Epoch {epoch}')  
        epoch_loss = 0.
        for clean, noisy in tqdm(train_dataloader):
            optimizer.zero_grad()
            x, y = clean.to(device), noisy.to(device)
            t = torch.rand(x.shape[0], device=x.device) * (sde.T - t_eps) + t_eps
            mean, std = sde.marginal_prob(x, y, t)
            z = torch.randn_like(x)
            sigma = std[:, None, None, None]
            x_t = mean + sigma * z
            forward_out = sde.forward(model, x_t, y, t)
            target = ((-1) * z) / sigma
            # print(f'forward_out requires grad: {forward_out.requires_grad}')
            # print(f'target requires grad: {target.requires_grad}')
            loss = criterion(forward_out, target)
            loss.backward()
            optimizer.step()
            ema.update()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        logging.info(f'Loss: {avg_loss}')  
        print(f'Loss: {avg_loss}')

        # Save the model with EMA parameters
        if epoch%10==0 and epoch!=0:
            with ema.average_parameters():
                torch.save(model.state_dict(), f'ckpts/{args.sde}_{epoch+80}.ckpt')

    

    # # Create an "enhanced" directory if it doesn't exist
    # os.makedirs('enhanced', exist_ok=True)

    # # PESQ Calculation
    # pesq_scores = []
    # # inverse_spec = InverseSpectrogram(n_fft=mel_transform.n_fft//2 + 1)

    # with torch.no_grad():
    #     for idx, (clean, noisy) in enumerate(tqdm(test_dataloader)):
    #         x, y = clean.to(device), noisy.to(device)
            
    #         # Generate enhanced mel spectrogram using predictor_corrector_sample
    #         enhanced_mel = sde.predictor_corrector_sample(model, shape=y.shape, device=device, y=y, n_steps=50)
    #         enhanced_mel = enhanced_mel.cpu().numpy()
    #         # print(enhanced_mel.squeeze(0).shape)
    #         # enhanced_mel = enhanced_mel.squeeze(0)
    #         # enhanced_mel = enhanced_mel.clamp(min=1e-8)
    #         # enhanced_mel = torch.nn.functional.softplus(enhanced_mel)
    #         # enhanced_mel = torch.sqrt(enhanced_mel)
    #         # print(enhanced_mel)
    #         # if torch.any(torch.isnan(enhanced_mel)) or torch.any(torch.isinf(enhanced_mel)):
    #         #     print("Enhanced mel contains NaNs or Infs")

    #         # Convert the mel spectrogram back to waveform
    #         enhanced_audio = librosa.feature.inverse.mel_to_audio(enhanced_mel.squeeze(), hop_length=507,sr=SAMPLE_RATE)
    #         y_in = y.cpu().numpy()
    #         y_in = librosa.feature.inverse.mel_to_audio(y_in.squeeze(), hop_length=507,sr=SAMPLE_RATE)
    #         # enhanced_audio = enhanced_audio.squeeze(0)

    #         # Save the enhanced audio file
    #         enhanced_audio_path = os.path.join('enhanced', f'enhanced_{idx}.wav')
    #         # enhanced_audio = torch.tensor(enhanced_audio)
    #         # torchaudio.save(enhanced_audio_path, enhanced_audio.cpu(), 16000)
    #         sf.write(enhanced_audio_path, enhanced_audio, SAMPLE_RATE)
            
    #         # Calculate PESQ score
    #         pesq = PerceptualEvaluationSpeechQuality(SAMPLE_RATE, 'wb')
    #         # pesq_score = pesq(y.cpu().numpy(), enhanced_audio.cpu().numpy())
    #         print(f'Length of clean audio: {y_in.shape}')
    #         print(f'Length of enhanced audio: {enhanced_audio.shape}')

    #         pesq_score = pesq(torch.tensor(y_in), torch.tensor(enhanced_audio))
    #         pesq_scores.append(pesq_score)
    #         logging.info(f'PESQ Score for sample {idx}: {pesq_score}')
    #         print(f'PESQ Score for sample {idx}: {pesq_score}')

    # # Log average PESQ score
    # avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0
    # logging.info(f'Average PESQ Score: {avg_pesq}')
    # print(f'Average PESQ Score: {avg_pesq}')
  

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--theta', type=int, default=1.5)
    parser.add_argument('--sde', type=str, default='ve', choices=['ve', 'vp', 'subvp'])
    parser.add_argument('--data_path', type=str, required=True, help="Path to the root directory of TIMIT dataset")
    args = parser.parse_args()
    main(args)
