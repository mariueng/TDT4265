import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

modules = list(model.children())
for i in range(len(modules) - 2):
    image = modules[i].forward(image)

image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]

# Task 4b
def visualize_filter_activations():
    # Create figure and axes
    plot_path = Path("plots")
    plot_path.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, len(indices), figsize=(20, 18))

    plt.setp(axs[0, :],
            xticks=[x for x in range(0, 6 + 1, 2)],
            yticks=[y for y in range(0, 6 + 1)])
    plt.setp(axs[1, :],
            xticks=[y for y in range(0, 101, 20)],
            yticks=[y for y in range(0, 101, 20)])

    # For each indice, retrieve weights and filters and add them to axis with the corresponding index.
    for i, indice in enumerate(indices):
        w = torch_image_to_numpy(first_conv_layer.weight[indice])
        f = torch_image_to_numpy(activation[0, indice]) 
        axs[0, i].imshow(w)
        axs[1, i].imshow(f, cmap="gray")

    plt.show()
    plt.savefig(plot_path.joinpath(f"Task4b.png"))


if __name__ == "__main__":
    visualize_filter_activations()
