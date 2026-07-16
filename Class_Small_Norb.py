import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import math



class SmallNORBPairDataset(Dataset):
    """
    Dataset de PARES para smallNORB.

    Para cada amostra devolve:
        image_A  : (1, H, W)  — imagem de origem
        image_B  : (1, H, W)  — imagem do mesmo objecto numa pose diferente
        T        : (2, 3)     — matriz afim 2 * 3 calculada a partir dos labels
        params   : (3,)       — [delta_azimuth_deg, delta_elevation_deg, 0.]

    A transformação T é calculada a partir da diferença de azimute e elevação
    entre image_A e image_B — não é aplicada artificialmente.
    """

    _FILES = {
        'train': {
            'dat':  'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
            'cat':  'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
            'info': 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
        },
        'test': {
            'dat':  'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
            'cat':  'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
            'info': 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
        }
    }


# trainset = SmallNORBPairDataset(
#             PATH_DATASET_NORB,
#             split='train',
#             image_size=32,
#             pair_mode='azimuth',
#             max_delta=1        # pares com diferença de 1 step = 20° de azimute
#         )
#         test_dataset  = SmallNORBPairDataset(PATH_DATASET_NORB, split='test',  image_size=32)
#         input_dim     = 32 * 32   # 1024 — grayscale 32×32

    def __init__(self, dataset_root, split='train', image_size=32,
                 pair_mode='azimuth', max_delta=1):
        """
        Args:
            dataset_root : str   — directório com os ficheiros .mat
            split        : str   — 'train' ou 'test'
            image_size   : int   — dimensão para resize (default: 32)
            pair_mode    : str   — 'azimuth', 'elevation', ou 'both'
            max_delta    : int   — diferença máxima de steps entre pares
                                   (1 step azimute = 20°, 1 step elevação = 5°)
        """
        assert split    in ('train', 'test')
        assert pair_mode in ('azimuth', 'elevation', 'both')

        self.dataset_root = dataset_root
        self.split        = split
        self.image_size   = image_size
        self.pair_mode    = pair_mode # 'azimuth', 'elevation', or 'both'. Qual quer par de imagens deve ter a mesma categoria, instância e iluminação, mas pode variar em azimute, elevação ou ambos.
        self.max_delta    = max_delta # Maximum difference in steps between pairs (1 step azimuth = 20°, 1 step elevation = 5°)

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(), # Converter imagens array em PIL para trabalhar em pytorch
            transforms.Resize((image_size, image_size)), # redimensionar para o tramanho desejado 
            transforms.ToTensor(),  # Converter imagens PIL em tensores PyTorch (C, H, W) e normalizar para [0, 1]
        ])

        # Carregar dados brutos
        self.images     = self._load_images()        # (N, 96, 96) uint8 Todas as imagens esquerdas
        self.categories = self._load_categories()    # (N,) todas as categorias
        self.info       = self._load_info()          # (N, 4): instance, elev, azim, light todas as informações de cada imagem que ajudam no calculo da pose T entre duas imagens. instance, elev, azim, light

        # Construir lista de pares válidos
        self.pairs = self._build_pairs()

    # ── Leitura dos ficheiros binários ────────────────────────────────────────

    def _path(self, key):
        return os.path.join(self.dataset_root, self._FILES[self.split][key])

    # lê o ficheiro .mat e retorna um array numpy contendo todas as dat, cat, info. O
    def _load_mat(self, filepath): 
        with open(filepath, 'rb') as f:
            magic = struct.unpack('<i', f.read(4))[0] # lê o número mágico do ficheiro .mat (4 bytes, little-endian), identifica o tipo de dados armazenados 
            ndim  = struct.unpack('<i', f.read(4))[0] # lê o número de dimensões do array (4 bytes, little-endian)
            dims  = [struct.unpack('<i', f.read(4))[0] for _ in range(max(ndim, 3))] # lê as dimensões do array (4 bytes cada, little-endian), garante que pelo menos 3 dimensões são lidas para compatibilidade com arrays 2D e 3D
            dims  = dims[:ndim] # corta a lista de dimensões para o número real de dimensões do array
            # dims = [24300, 2, 96, 96]

            # Mapeamento correcto conforme especificação NYU
            dtype_map = {
                0x1E3D4C51: np.float32,   # single precision
                0x1E3D4C52: np.uint8,     # packed
                0x1E3D4C53: np.float64,   # double precision
                0x1E3D4C54: np.int32,     # integer
                0x1E3D4C55: np.uint8,     # byte matrix 
                0x1E3D4C56: np.int16,     # short
            }
            if magic not in dtype_map:
                raise ValueError(f"Magic number desconhecido: {hex(magic)}")
            dtype = dtype_map[magic]

            data = np.frombuffer(f.read(), dtype=dtype).reshape(dims) # lê os dados restantes do ficheiro e converte para um array numpy com o tipo de dados correto e as dimensões apropriadas
            # data.shape([24300, 2, 96, 96])
        return data

    def _load_images(self):
        raw = self._load_mat(self._path('dat'))
        return raw[:, 0, :, :]   # câmara esquerda — (N, 96, 96)

    def _load_categories(self):
        return self._load_mat(self._path('cat')).flatten()

    def _load_info(self):
        return self._load_mat(self._path('info'))   # (N, 4)

    # ── Construção de pares ───────────────────────────────────────────────────

    def _build_pairs(self):
        """
        Agrupa amostras por (categoria, instância, iluminação) e constrói
        pares (idx_A, idx_B) com delta de azimute ou elevação controlado.

        Dois exemplos formam um par válido se:
          - mesma categoria
          - mesma instância 
          - mesma iluminação de 0 a 5, tipos de iluminação 
          - delta de azimute ou elevação dentro de max_delta steps # Maximum difference in steps between pairs (1 step azimuth = 20°, 1 step elevation = 5°)
        """
        pairs = []

        # Agrupar índices por (categoria, instância, iluminação)
        groups = {}
        # print(len(self.categories)) 24300
        for i in range(len(self.categories)):
            cat      = int(self.categories[i])
            instance = int(self.info[i, 0])
            # azimuth = int(self.info[i, 1])
            # elevation = int(self.info[i, 2])
            lighting = int(self.info[i, 3])
            # key = (cat, instance, elevation, lighting) 
            key = (cat, instance, lighting) 
            groups.setdefault(key, []).append(i)

        for key, indices in groups.items():
            for i_pos, idx_A in enumerate(indices):
                elev_A = int(self.info[idx_A, 1])
                azim_A = int(self.info[idx_A, 2])

                for idx_B in indices[i_pos + 1:]:
                    elev_B = int(self.info[idx_B, 1])
                    azim_B = int(self.info[idx_B, 2])

                    delta_azim = abs(azim_B - azim_A)
                    delta_elev = abs(elev_B - elev_A)

                    if self.pair_mode == 'azimuth':
                        # Apenas azimute varia, elevação igual
                        # if delta_elev == 0 and 0 < delta_azim <= self.max_delta * 2:
                        pairs.append((idx_A, idx_B))

        return pairs

    # ── Construção da matriz T ────────────────────────────────────────────────

    def _build_T(self, idx_A, idx_B):
        """
        Constrói a matriz afim 2×3 a partir da diferença de labels.

        O azimute é mapeado para rotação 2D no plano da imagem.
        A elevação é mapeada para translação vertical normalizada.

        Returns:
            T      : torch.Tensor (2, 3)
            params : torch.Tensor (3,)  — [delta_azim_deg, delta_elev_deg, 0.]
        """
        elev_A = int(self.info[idx_A, 1])
        azim_A = int(self.info[idx_A, 2])
        elev_B = int(self.info[idx_B, 1])
        azim_B = int(self.info[idx_B, 2])

        # Converter para valores reais
        delta_azim_deg = (azim_B - azim_A) * 20.0    # graus
        delta_elev_deg = (elev_B - elev_A) * 5.0     # graus

        # Azimute → rotação no plano da imagem
        theta_rad = delta_azim_deg * (math.pi / 180.0)
        cos_t     = math.cos(theta_rad)
        sin_t     = math.sin(theta_rad)

        # Elevação → translação vertical normalizada [-1, 1]
        # Intervalo de elevação total: 40° (de 30° a 70°)
        dy_norm = delta_elev_deg / 40.0

        # Matriz 2×3:
        #   | cos(θ)  -sin(θ)   0     |
        #   | sin(θ)   cos(θ)   dy    |
        T = torch.tensor([
            [cos_t, -sin_t, 0.0   ],
            [sin_t,  cos_t, dy_norm]
        ], dtype=torch.float32)

        params = torch.tensor(
            [delta_azim_deg, delta_elev_deg, 0.0],
            dtype=torch.float32
        )

        return T, params

    # ── Interface Dataset ─────────────────────────────────────────────────────

    def __len__(self): # return the number of pairs in the dataset
        return len(self.images)
        # return len(self.pairs)

    def __getitem__(self, idx): # return a pair of images and the corresponding transformation matrix T
        # idx_A, idx_B = self.pairs[idx]
        
        image = self.base_transform(self.images[idx])  # (1, 32, 32)

        return image
        # image_A = self.base_transform(self.images[idx_A])  # (1, H, W)
        # image_B = self.base_transform(self.images[idx_B])  # (1, H, W)

        # T, params = self._build_T(idx_A, idx_B)

        #return image_A, image_B, T, params