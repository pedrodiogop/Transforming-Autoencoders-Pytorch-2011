import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, img_c, img_h, img_w, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.img_c = img_c
        self.img_h = img_h
        self.img_w = img_w
        
        self.image_filenames = [ f for f in os.listdir(folder_path) ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        # Carrega a imagem. L convert para Grayscale (1 canal), RGB para 3 canais
        if self.img_c == 1:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
            
        # Redimensiona para o tamanho que o teu modelo espera (ex: 28x28 ou 32x32)
        image = image.resize((self.img_w, self.img_h))

        if self.transform:
            image = self.transform(image)

        return image, 0
    