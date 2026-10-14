import os
import re
import torch
from collections import defaultdict
from itertools import product
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Gramática oficial: person[Id(2)][Serie(1)][Number(2)][Tilt(assinado)][Pan(assinado)].jpg
# ex: person08123-30+45.jpg -> Id=08, Serie=1, Number=23, Tilt=-30, Pan=+45
FILENAME_PATTERN = {
    'train': re.compile(
        r'^person(\d{2})1(\d{2})([+-]\d{1,2})([+-]\d{1,2})\.jpg$',
        re.IGNORECASE
    ),
    'test': re.compile(
        r'^person(\d{2})2(\d{2})([+-]\d{1,2})([+-]\d{1,2})\.jpg$',
        re.IGNORECASE
    )
}

class Class_Pointing04(Dataset):
    # def __init__(self, path_file, transform=None, img_c=3, img_w=384, img_h=288):
    def __init__(self, path_file, img_color, img_w, img_h, split):

        assert split in FILENAME_PATTERN

        self.path_file = path_file
        self.split = split  # 'train' ou 'test'
                
        self.base_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),  # (height, width)
            transforms.ToTensor(),  # PIL → Tensor (C, H, W), normaliza [0, 1]
        ])


        # groups: (person_id) -> lista de (path, tilt, pan)
        groups = defaultdict(list)
        for fname in os.listdir(path_file):
            match = FILENAME_PATTERN[self.split].match(fname)
            if match is None:
                continue  # ignora ficheiros que não encaixam na gramática person[Id][Serie][Number][Tilt][Pan].jpg
            person_id, _, tilt, pan = match.groups()
            groups[person_id].append((os.path.join(path_file, fname), int(tilt), int(pan)))

        if not groups:
            raise RuntimeError(f"Nenhuma imagem reconhecida em {path_file}.")

        for key, items in groups.items():
            if len(items) != 93:
                print(f"Aviso: grupo {key} tem {len(items)} imagens (esperado 93).")

        # 2. Cache em memória: cada imagem é decodificada uma única vez (uint8)
        self.images, index = [], {}
        for items in groups.values():
            for path, _, _ in items:
                img = Image.open(path).convert(img_color)
                t = (self.base_transform(img) * 255).round().to(torch.uint8)
                index[path] = len(self.images)
                self.images.append(t)

        # 2. Produto cartesiano ordenado dentro de cada (pessoa, série)
        self.pairs = []
        for _, items in groups.items():
            for (path_a, tilt_a, pan_a), (path_b, tilt_b, pan_b) in product(items, repeat=2):
                pose_diff = [(tilt_b - tilt_a) / 180, (pan_a - pan_b) / 180 ]  # normalizado para [-1, 1]
                self.pairs.append((index[path_a], index[path_b], pose_diff))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ia, ib, diff = self.pairs[idx]
        a = self.images[ia].float() / 255
        b = self.images[ib].float() / 255
        return a, b, torch.tensor(diff, dtype=torch.float32) # diff -> (tilt_diff, pan_diff) normalizado para [-1, 1]