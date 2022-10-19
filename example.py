from cifar_10_100 import load_CIFAR
import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, title):
    fig, axs = plt.subplots(nrows=4, ncols=5,
                            subplot_kw={'xticks': [], 'yticks': []},
                            # figsize=(23.4, 12.),
                            num=title,
                            )
    size = 1024  # image pixel size is 1024=32*32
    for n, ax in enumerate(axs.flat):
        # Extract RGB for each image and reshape them to 2D 32x32 array
        image = images[n]
        r = image[0:size].reshape(32, 32)
        g = image[size:size * 2].reshape(32, 32)
        b = image[size * 2:size * 3].reshape(32, 32)
        # Consolidate rgb to form 3D RGB image
        image = np.stack([r, g, b], axis=2)  # shape is (32x32x3)
        ax.imshow(image, cmap='viridis', interpolation='antialiased')
    plt.subplots_adjust(top=0.990, bottom=0.010, left=0.010, right=0.990,
                        wspace=0.150, hspace=0.150)

if __name__ == '__main__':
    # Get datasets
    cifar10 = load_CIFAR(10)
    cifar100 = load_CIFAR(100)
    # Get training images
    images10 = cifar10['train']['images'].pixels
    images100 = cifar100['train']['images'].pixels
    # Plot images
    plot_images(images10, "CIFAR-10 first 20 Training Images")
    plot_images(images100, "CIFAR-100 first 20 Training Images")
    plt.show()
