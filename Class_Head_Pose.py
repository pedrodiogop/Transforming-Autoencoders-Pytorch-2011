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
FILENAME_PATTERN = re.compile(
    r'^person(\d{2})(\d)(\d{2})([+-]\d{1,2})([+-]\d{1,2})\.jpg$',
    re.IGNORECASE
)

class HeadPoseDataset(Dataset):
    def __init__(self, path_file, transform=None, img_c=3, img_w=384, img_h=288):
        self.path_file = path_file
        self.transform = transform
        self.img_c = img_c
        self.img_h = img_h  # altura (height)
        self.img_w = img_w  # largura (width)

                
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),  # (height, width)
            transforms.ToTensor(),  # PIL → Tensor (C, H, W), normaliza [0, 1]
        ])

        # 1. Scan recursivo + parsing do nome do ficheiro
        # groups: (person_id, serie_id) -> lista de (path, tilt, pan)
        groups = defaultdict(list)
        for fname in os.listdir(path_file):
            match = FILENAME_PATTERN.match(fname)
            if match is None:
                continue  # ignora ficheiros que não encaixam na gramática person[Id][Serie][Number][Tilt][Pan].jpg
            person_id, serie_id, _, tilt, pan = match.groups()
            key = (person_id, serie_id)
            full_path = os.path.join(path_file, fname)
            groups[key].append((full_path, int(tilt), int(pan)))

        if not groups:
            raise RuntimeError(
                f"Nenhuma imagem reconhecida em {path_file}. "
                f"Verifica a estrutura de pastas e se os nomes seguem a gramática person[Id][Serie][Number][Tilt][Pan].jpg"
            )

        # (opcional, recomendado) valida que cada série tem exatamente 93 imagens
        for key, items in groups.items():
            if len(items) != 93:
                print(f"Aviso: grupo {key} tem {len(items)} imagens (esperado 93).")

        # 2. Produto cartesiano ordenado dentro de cada (pessoa, série)
        self.pairs = []
        for key, items in groups.items():
            for (path_a, tilt_a, pan_a), (path_b, tilt_b, pan_b) in product(items, repeat=2):
                pose_diff = ( (tilt_b - tilt_a) / 90, (pan_b - pan_a) / 90 )
                self.pairs.append((path_a, path_b, pose_diff))

    def __len__(self):
        return len(self.pairs)
    
    def _load(self, path):
        image = Image.open(path).convert('RGB')
        image = self.base_transform(image)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        path_a, path_b, pose_diff = self.pairs[idx]
        image1 = self._load(path_a)
        image2 = self._load(path_b)
        pose_diff = torch.tensor(pose_diff, dtype=torch.float32)
        return image1, image2, pose_diff