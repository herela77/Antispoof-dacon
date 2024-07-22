import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

def pad2d(image, new_width):
    height, current_width = image.shape
    
    if current_width < new_width:
        expansion_width = new_width - current_width
        repeats = new_width // current_width
        extra = new_width % current_width
        
        full_part = np.tile(image, (1, repeats))
        extra_part = image[:, :extra]
        new_image = np.hstack((full_part, extra_part))
    
    elif current_width > new_width:
        start_index = np.random.randint(0, current_width - new_width)
        new_image = image[:, start_index:start_index + new_width]
    
    else:
        new_image = image
    
    return new_image

def process_audio_file(args):
    path, sr, n_bins, hop_length, bins_per_octave = args
    y, _ = librosa.load(path, sr=sr)
    cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_db = (cqt_db - cqt_db.mean()) / cqt_db.std()
    cqt_db = pad2d(cqt_db, 100)
    return cqt_db

def process_label(label, n_classes):
    label_vector = np.zeros(n_classes, dtype=float)
    if isinstance(label, tuple):
        label_vector = [label[0], label[1]]
    else:
        label_vector[0 if label == 'fake' else 1] = 1
    return label_vector

class AudioDataset(Dataset):
    def __init__(self, df, sr=32000, n_bins=84, bins_per_octave=12, hop_length=512, n_classes=2, train_mode=True, transform=None, num_workers=100):
        self.sr = sr
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.n_classes = n_classes
        self.train_mode = train_mode
        self.num_workers = num_workers

        self.audio_paths = df['path'].tolist()
        self.labels = df['label'].tolist() if train_mode else None

        # 멀티프로세싱으로 오디오 파일 전처리
        with mp.Pool(num_workers) as pool:
            args = [(path, sr, n_bins, hop_length, bins_per_octave) for path in self.audio_paths]
            self.audio_data = list(tqdm(pool.imap(process_audio_file, args), total=len(args)))

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio = self.audio_data[index]

        if self.train_mode:
            label = self.labels[index]
            label_vector = process_label(label, self.n_classes)
            return torch.tensor(audio, dtype=torch.float32), torch.tensor(label_vector, dtype=torch.float32)

        return torch.tensor(audio, dtype=torch.float32)
