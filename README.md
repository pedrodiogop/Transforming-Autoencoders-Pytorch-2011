# Transforming-Autoencoders-Pytorch-2011

This project is based on the research presented in the following paper:

    [Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 2011.](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf)

The codebase was originally forked from [IsCoelacanth](https://github.com/IsCoelacanth/TransformingAutoencoder_PyTorch)

## Key Modifications:
*   Added command-line arguments 
*   Multi-Dataset Support: The architecture has been adapted to handle different datasets beyond the original MNIST.
*   Capsule Activation Functions: Integrated non-linear activation functions within the capsules to improve feature representation and gradient flow.
*   Replaced the `Sigmoid` + `BCELoss` combination with `BCEWithLogitsLoss` for more stable gradient calculation.
*   Results are dynamically saved in a hierarchical folder structure based on the hyperparameters: Results/{dataset}/{batch_size}_{num_caps}_{cap_rec}_{cap_gen}_{learning_rate}/. Example: Results/fashion-mnist/6_25_16_16_0.001/. Inside each folder we store: 'Image_Loss' -> A plot representing the Loss Function evolution for each epoch.
* **Gradient Analysis Plots:**
  * `Gradients_Per_Capsule`: Visualizes the mean gradient behavior for each individual capsule across epochs or batch_size.
  * `Gradients_Per_Layer`: Displays the gradient flow across different layers across epoch or batch_size.



## Code is divided into: 
    main.py -> Model training
    capLayer.py -> Capsule layer
    capsule.py -> Individual capsule
    aux_functions.py -> auxiliary functions
    test.py -> Model testing 


## Command-Line Arguments (default config):
    --device -> mps (macbook)
    --batch_size -> 64
    --epoch -> 15
    --num_caps -> 25 (Number of capsules)
    --cap_rec -> 40 (encode)
    --gen_dim -> 40 (decode)
    --lr -> 0.001
    --dataset -> MNIST, also accepts 'FashionMNIST' and 'CIFAR10'

## Running the code: 
    python3 main.py 


    # 1. Garantir que a pasta 'plots' existe
    os.makedirs('images/plots', exist_ok=True)
    # 2. Criar a figura
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(loss_history)), loss_history, label=f'Função de Custo')
    # 4. Estilizar o gráfico
    ax.set_title('Função de Custo')
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Loss (BCELoss)')
    ax.grid(True)
    ax.legend()
    # 5. Guardar a imagem
    caminho_plot = 'images/plots/funcao_custo.png'
    fig.savefig(caminho_plot)    
    # 6. Fechar a figura para libertar memória RAM
    plt.close(fig)


