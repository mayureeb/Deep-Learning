Deep Learning Experiments (PyTorch)
==================================

This bundle has a few self-contained PyTorch scripts that can be dropped into
your `Deep-Learning` repository and run locally. They cover both basic and
more advanced / slightly novel ideas.

Files
-----
1. `01_mnist_mlp.py`
   - Classic fully connected network on MNIST.
   - Shows basic training loop, accuracy tracking and saving weights.

2. `02_cifar10_cnn.py`
   - Compact CNN for CIFAR-10.
   - Has a flag `use_augmentation` which switches data augmentation on/off so
     you can demonstrate the effect of augmentation on test accuracy by
     running the script twice and comparing results.

3. `03_mnist_denoising_autoencoder.py`
   - Denoising autoencoder: the model gets a noisy digit and learns to
     reconstruct the clean version.
   - After training it saves a figure
     `figures/mnist_denoising_examples.png` with three rows:
     clean / noisy / denoised digits. This is a nice visual result to show
     on GitHub or slides.

4. `04_cifar10_resnet_transfer.py`
   - Transfer learning on CIFAR-10 using ImageNet-pretrained ResNet-18.
   - First does a **linear probe** (only the final layer is trained) and
     prints the test accuracy.
   - Then unfreezes all layers and continues with a small learning rate to
     do **full fine-tuning**, saving separate weights for each stage.

5. `requirements.txt`
   - Minimal dependencies (`torch`, `torchvision`, `matplotlib`).

How to Run
----------
```bash
pip install -r requirements.txt

# 1) Simple MNIST MLP
python 01_mnist_mlp.py

# 2) CIFAR10 CNN + augmentation
python 02_cifar10_cnn.py

# 3) Denoising autoencoder with visual outputs
python 03_mnist_denoising_autoencoder.py

# 4) Transfer learning with ResNet18
python 04_cifar10_resnet_transfer.py
```

You can then commit the scripts, the requirements file, and a sample
`figures/mnist_denoising_examples.png` image plus some training logs to your
existing `Deep-Learning` repo to clearly show both basic understanding and
more advanced topics like data augmentation, denoising autoencoders and
transfer learning.
