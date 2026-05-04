from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from image_utils import BatchShift, BatchShift_torch, Show_Batch
import torch
import torchvision
from CapLayer import CapLayer
import matplotlib.pyplot as plt
import torch.nn as nn

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    BATCH_SIZE = 124
    NUM_CAPS = 25
    IN_DIM = 28 * 28
    CAP_REC = 40
    GEN_DIM = 40    

    testset = datasets.MNIST(
    root="tmp",
    train=False,
    download=True,
    transform=ToTensor() # Transforma as imagens em tensores PyTorch, normalizando os valores dos pixels para o intervalo [0, 1], shape (canal, altura, largura).
)
    testeloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, GEN_DIM)
    capL_test = capL_test.to(device)
    #print(capL_test)

    # 2. Carregar os pesos do ficheiro
    capL_test.load_state_dict(torch.load("best_checkpoint.pth", map_location=device))
    crit = nn.BCELoss() # Binary Cross entropy Loss, vai comparar a imagem reconstruida com a imagem original.

    # 3. Colocar em modo de avaliação (importante!)
    capL_test.eval()
    test_loss = 0.0
    with torch.no_grad(): # Desliga gradientes para poupar memória
        for img, _ in testeloader:
            img = img.to(device)
            #targ_np, dxy_np = BatchShift(img.numpy().copy(), [-4,4])
            targ_np, dxy_np = BatchShift_torch(img, [-4, 4], device)   # tudo na GPU
            
            #img, dxy = img.to(device), torch.from_numpy(dxy_np).float().to(device)
            #target = torch.from_numpy(targ_np).float().to(device)
            
            out = capL_test(img, dxy_np)
            loss = crit(out.view(-1, 1, 28, 28), targ_np)
            test_loss += loss.item()

    print(f"Loss Média no Teste: {test_loss/len(testeloader):.4f}")
    #Showing test images
    img, _ = next(iter(testeloader))

    targ , dxy = BatchShift(img.numpy().copy(),[-4,4])
    img = img.to(device)
    dxy = torch.from_numpy(dxy).float().to(device)    
    out, R = capL_test(img, dxy, sep=True)
    out = out.view(-1,1,28,28)
    Show_Batch(img, torch.from_numpy(targ),out, epoch = 11, i = 1, title='Test_Input', save=True)
    plt.show()
    #print(R)