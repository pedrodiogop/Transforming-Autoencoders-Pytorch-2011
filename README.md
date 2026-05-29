# Transforming-Autoencoders-Pytorch-2011

This project is based on the research presented in the following paper:

    [Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 2011.](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf)

Adicionar uma breve introdução ao paper, mostrar brevemente os resultados, e etc...

## Requirements

## Usage

### The Default Hyper Parameters:
| CLI Arguments | Value | Help |
| --- | --- | --- | 
| --device | mps | Device to use for training (e.g., "cpu", "cuda", "mps") Never tested for cuda |
    --device -> mps (macbook)
    --batch_size -> 64
    --epoch -> 15
    --num_caps -> 25 (Number of capsules)
    --cap_rec -> 40 (encode)
    --gen_dim -> 40 (decode)
    --lr -> 0.001
    --dataset -> MNIST, also accepts 'FashionMNIST' and 'CIFAR10'
    --len_pose, if --len_pose = 2 we train the model to Equivariance Mode (with displacement). If --len_pose > 2 we train the model to Generative Reconstruction Mode (only to reconstruction the image, without displacement).

## Results 

## Model Design

## Credits

## Key Modifications:
*   Added command-line arguments 
*   Multi-Dataset Support: The architecture has been adapted to handle different datasets beyond the original MNIST.
*   Capsule Activation Functions: Integrated non-linear activation functions within the capsules to improve feature representation and gradient flow.
*   Replaced the `Sigmoid` + `BCELoss` combination with `BCEWithLogitsLoss` for more stable gradient calculation.
*   Results are dynamically saved in a hierarchical folder structure based on the hyperparameters: Results/{dataset}/{batch_size}_{num_caps}_{cap_rec}_{cap_gen}_{learning_rate}/. Example: Results/fashion-mnist/6_25_16_16_0.001/. Inside each folder we store: 'Image_Loss' -> A plot representing the Loss Function evolution for each epoch.
*   Function that save the Input/Target/Output Images for comparison
* **Gradient Analysis Plots:**
  * `Gradients_Per_Capsule`: Visualizes the mean gradient behavior for each individual capsule across epochs or batch_size.
  * `Gradients_Per_Layer`: Displays the gradient flow across different layers across epoch or batch_size.
* **Latent Space & Spatial Equivariance Diagnostics (`poses.py`):** Developed a dedicated evaluation framework to validate the network's mathematical behavior against Geoffrey Hinton's core formulation of capsule-based coordinate frames.
    * **Methodology:** Tracks and extracts the learned internal capsule pose parameters before and after applying physical horizontal shifts generated dynamically via PyTorch `affine_grid` and `grid_sample` mapping.
    * **Linear Equivariance Validation:** Outputs comprehensive scatter plots with linear regression lines, proving a high correlation between original and displaced latent representations, backed by robust Coefficients of Determination.
    * **Directional Awareness:** Visualizes a distinct, parallel geometric separation between the **Less (-3)** and **More (+3)** translation trends, confirming that the targeted capsule successfully internalizes both the magnitude and direction of the displacement vector ($dxy$).
    * **Feature Disentanglement:** Demonstrates that the architecture successfully extracts a continuous coordinate system for spatial reasoning instead of merely memorizing surface-level pixel intensities.
    * Within the Poses/Comparison_Original_Shift folder, we can see some results filtered by the probability of the capsule(threshold >= 0.9), removing some noise.
    * Below is the generated scatter plot proving the linear relationship of the latent space (): ![Pose Equivariance Plot](Poses/ALL_Data_Poses_Capsule_2_Combined_[LessMore,Original]_Comparisons.png)
* **Dual-Mode Training Strategy: Equivariance vs. Reconstruction Capacity**: The architecture is highly flexible and dynamically adapts its training behavior based on the `--len_pose` hyperparameter:
    * **Equivariance Mode (`--len_pose 2`):** Restricts the latent space strictly to $X$ and $Y$ coordinates to enforce and analyze geometric translation equivariance.
    * **Generative Reconstruction Mode (`--len_pose > 2`):** Bypasses the spatial displacement constraint to maximize the model's capacity for complex image synthesis. Used for training with the cifar10 dataset. 
* **Evaluation**: To support the dual-mode architecture strategy, the framework includes two dedicated testing scripts, one for image reconstruction other for equivariance.



## Code is divided into: 
    main.py -> Model training
    capLayer.py -> Capsule layer
    capsule.py -> Individual capsule
    aux_functions.py -> auxiliary functions
    gradients_aux.py -> gradient functions  
    test.py -> Model testing
    poses.py -> To trace the relationship between poses in the displaced images and the original images.

## Running the code: 
    python3 main.py 
    python3 poses.py



The codebase was originally forked from [IsCoelacanth](https://github.com/IsCoelacanth/TransformingAutoencoder_PyTorch)


