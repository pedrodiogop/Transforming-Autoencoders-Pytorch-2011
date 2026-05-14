# Transforming-Autoencoders-Pytorch-2011

This project is based on the research presented in the following paper:

Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 2011. [http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf]

The codebase was originally forked from [IsCoelacanth](https://github.com/IsCoelacanth/TransformingAutoencoder_PyTorch)

# Code is divided into: 
    # main.py -> Model training
    # capLayer.py -> Capsule layer
    # capsule.py -> Individual capsule
    # images_utils.py -> auxiliary functions like translation and saving images
    # test.py -> Model testing 


# Default Config:
    --device -> mps (macbook)
    --batch_size -> 64
    --epoch -> 15
    --num_caps -> 25 (Number of capsules)
    --cap_rec -> 40 (encode)
    --gen_dim -> 40 (decode)
    --lr -> 0.001

# Running the code: 
    python3 main.py 



