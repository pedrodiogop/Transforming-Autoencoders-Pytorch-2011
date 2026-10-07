import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import math



class SmallNORBPairDataset(Dataset):

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

    def __init__(self, dataset_root, split, image_size, crop_size):

        assert split    in ('train', 'test')

        self.dataset_root = dataset_root
        self.split        = split
        self.image_size   = image_size
        self.crop_size    = crop_size  # tamanho original das imagens SmallNORB

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(), # Converter imagens array em PIL para trabalhar em pytorch
            transforms.CenterCrop(self.crop_size),
            transforms.Resize((image_size, image_size)), # redimensionar para o tramanho desejado 
            transforms.ToTensor(),  # Converter imagens PIL em tensores PyTorch (C, H, W) e normalizar para [0, 1]
        ])

        # Carregar dados brutos
        self.l_images     = self._load_l_images()    # 24300x96x96
        self.r_images     = self._load_r_images()    # 24300x96x96
        self.categories = self._load_categories()    # (N,) todas as categorias
        self.info       = self._load_info()          # (N, 4): instance, elev, azim, light

        # self.info_categoria_l, self.info_categoria_r  = self._pre_processemnt_join_img_labels()
        
        self._pre_processemnt_join_img_labels()

        # Construir lista de pares válidos
        # self.pairs = self._build_pairs()

    # ── Leitura dos ficheiros binários ────────────────────────────────────────
    def _path(self, key):
        return os.path.join(self.dataset_root, self._FILES[self.split][key])

    # lê o ficheiro .mat e retorna um array numpy contendo todas as dat, cat, info
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

    def _load_l_images(self):
        raw = self._load_mat(self._path('dat'))
        return raw[:, 0, :, :] # 24300x96x96

    def _load_r_images(self):
        raw = self._load_mat(self._path('dat'))
        return raw[:, 1, :, :] # 24300x96x96

    def _load_categories(self):
        return self._load_mat(self._path('cat')).flatten()

    def _load_info(self):
        return self._load_mat(self._path('info'))   # (N, 4)

    def _pre_processemnt_join_img_labels(self):
        categorias = self.categories
        info = self.info
        l_images = self.l_images
        # r_images = self.r_images

        # info_categoria: [instancia, categoria, ele, azim, ilum]
        info_categoria = np.insert(info, 1, categorias, axis=1)
        info_categoria = info_categoria.astype(np.float32)

        # normalização da elevação e do azimute
        info_categoria[:, 2] = ((info_categoria[:, 2] + 30 + info_categoria[:, 2] * 4) - 50) / 25
        info_categoria[:, 3] = ((info_categoria[:, 3] * 10) - 160) / 200

        # remove a última coluna (ilum) -> [instancia, categoria, ele, azim]
        info_categoria = np.delete(info_categoria, -1, axis=1)

        # # adiciona o índice global de cada imagem -> [instancia, categoria, ele, azim, idx]
        # idx = np.arange(info_categoria.shape[0])
        # info_categoria = np.column_stack((info_categoria, idx))

        info_categoria_l = info_categoria.copy()
        # info_categoria_r = info_categoria.copy()

        # valor do primeiro pixel de cada imagem, normalizado
        pixel_images_l = l_images[:, 0, 0] / 255.0
        # pixel_images_r = r_images[:, 0, 0] / 255.0

        # insere o pixel na coluna 4
        info_categoria_l = np.insert(info_categoria_l, 4, pixel_images_l, axis=1)
        # info_categoria_r = np.insert(info_categoria_r, 4, pixel_images_r, axis=1)

        # colunas de info_categoria_l: [ins, categ, ele, azim, pixel, idx]

        # ---------- agrupamento por (categoria, instância) ----------
        ins   = info_categoria_l[:, 0]        # (24300,) instância de cada imagem
        categ = info_categoria_l[:, 1]        # (24300,) categoria de cada imagem

        # (24300, 2): uma linha (categ, ins) por imagem
        pares_grupo = np.stack([categ, ins], axis=1)
        # print(pares_grupo[:20])
        # print(pares_grupo.shape) # (24300, 2)

        # número do grupo (0..24) de cada imagem
        _, grupo = np.unique(pares_grupo, axis=0, return_inverse=True)
        # [ 3  8 10 17 23  3  9 12 17 21  1  6 11 19 23  1  6 10 16 21  0  7 10 16 20  1  8 12 15 22  0]
        # print(grupo[:31])
        # print(grupo.shape) (24300,)
        # unique/return_inverse -> The indices from the set of unique elements that reconstruct x. Ou seja devolve o indice do elemento no array unique
        grupo = grupo.ravel()

        # índices das imagens ordenados por grupo
        # argsort -> retorna os indice que ordenava o array 
        order = np.argsort(grupo, kind="stable")
        # print(order.shape) (24300,)
        # [ 20  30  80  90 100 125 135 145 160 165 180 190 210 215 250 265 320 340 370 390]
        # print(order[:20]) 
        # Os primeiros 972 são as imagens do grupo 0, os 972 seguintes do grupo 1, e assim por diante

        # print(grupo.max()) 24
        self.n_grupos = int(grupo.max()) + 1                 # 25
        self.n_por_grupo = len(order) // self.n_grupos       # 972 -> Existem 972 imagens num grupo(instancia e categoria)

        # (25, 972): linha g = índices globais das imagens do grupo g
        self.group_idx = order.reshape(self.n_grupos, self.n_por_grupo)
        # print(self.group_idx.shape) (25, 972)

        # atributos (ele, azim, pixel) de cada imagem
        self.attrs_l = info_categoria_l[:, 2:5].astype(np.float32)   # (24300, 3)
        # self.attrs_r = info_categoria_r[:, 2:5].astype(np.float32)   # (24300, 3)

    def __len__(self): # return the number of pairs in the dataset
        return self.n_grupos * self.n_por_grupo ** 2
        # return len(self.pairs)

    def __getitem__(self, k):
            n = self.n_por_grupo # 972
            # divmod (x // y, x % y)
            # explicação de como o k restringe os valores de "g", "a" e "b" nas notas
            g, r = divmod(k, n * n)
            a, b = divmod(r, n)
            i = self.group_idx[g, a]
            t = self.group_idx[g, b]

            transf_l = self.attrs_l[t] - self.attrs_l[i]
            # transf_r = self.attrs_r[t] - self.attrs_r[i]
            return (self.base_transform(self.l_images[i]), self.base_transform(self.l_images[t]), transf_l)
            # return (self.base_transform(self.l_images[i]), self.base_transform(self.l_images[t]), self.base_transform(self.r_images[i]), self.base_transform(self.r_images[t]), transf_l, transf_r)

    # Codigo antes de Claude
    # def _pre_processemnt_join_img_labels(self):
    #     # na ordenação da Lighting ver primeiro quais numeros sao os mais claros e escuros 
    #     categorias = self.categories
    #     info = self.info
    #     l_images = self.l_images
    #     r_images = self.r_images

    #     info_categoria = np.insert(info, 1, categorias, axis=1) 
    #     # info_categoria [intancia, categoria, ele, azh, ilum]
    #     info_categoria = info_categoria.astype(np.float32)
    #     info_categoria[:, 2] = ((info_categoria[:, 2] + 30 + info_categoria[:, 2] * 4) - 50) / 25
    #     info_categoria[:, 3] = ((info_categoria[:, 3] * 10) - 160) / 200
    #     info_categoria = np.delete(info_categoria, -1, axis=1) # --> returns new array

    #     idx = np.arange(info_categoria.shape[0])  # 0.0, 1.0, 2.0, ...
    #     info_categoria = np.column_stack((info_categoria, idx))
        
    #     info_categoria_l = info_categoria.copy()
    #     info_categoria_r = info_categoria.copy()

    #     pixel_images_l = l_images[:, 0, 0] / 255.0
    #     pixel_images_r = r_images[:, 0, 0] / 255.0

    #     info_categoria_l = np.insert(info_categoria_l, 4, pixel_images_l, axis=1) 
    #     info_categoria_r = np.insert(info_categoria_r, 4, pixel_images_r, axis=1) 

    #     v_idx_transf = [] # shape = (24300, 5)
    #     for ins_i, categ_i, ele_i, azim_i, light_i, idx_i in info_categoria_l:
    #         for ins_t, categ_t, ele_t, azim_t, light_t, idx_t in info_categoria_l: 
    #             if ins_i == ins_t and categ_i == categ_t:
    #                 v_idx_transf.append([
    #                     idx_i, 
    #                     idx_t, 
    #                     ele_t - ele_i, 
    #                     azim_t - azim_i , 
    #                     light_t - light_i 
    #                     ])
    #     v_idx_transf = np.array(v_idx_transf)
    #     print(len(v_idx_transf)) # 23 619 600
    #     print(v_idx_transf.shape) # (23 619 600, 5)

    #     # # chaves: da menos importante para a mais importante
    #     # ordem = np.lexsort((info_categoria[:, 3],   # 4º critério
    #     #     info_categoria[:, 2],   # 3º critério
    #     #     info_categoria[:, 1],   # 2º critério
    #     #     info_categoria[:, 0]))  # 1º critério (principal)
    #     # info_categoria = info_categoria[ordem]

    #     return info_categoria_l, info_categoria_r