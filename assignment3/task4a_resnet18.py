import torchvision
from pathlib import Path
from torch import nn
import matplotlib.pyplot as plt

import utils
from trainer import Trainer, compute_loss_and_accuracy
from dataloaders import load_cifar10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax, as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
    
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Set seed
    utils.set_seed(0)

    # Set hyperparameters according to Task
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4

    # Load dataset and resize image while also setting the mean and std
    dataloaders = load_cifar10(
        batch_size,
        size=224,
        mean=(0.485, 0.456, 0.406),
        std= (0.229, 0.224, 0.225))

    # Instantiate model
    resnet18 = Model()

    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        resnet18,
        dataloaders,
        use_adam=True
    )
    trainer.train()

    # Plot and save
    plot_path = Path("plots")
    plot_path.mkdir(exist_ok=True)
    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.show()
    plt.savefig(plot_path.joinpath(f"Task4a_resnet18_plot.png"))

    # Retrieve best model
    trainer.load_best_model()
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_acc = compute_loss_and_accuracy(dataloader_train, trainer.model, nn.CrossEntropyLoss())
    _, val_acc = compute_loss_and_accuracy(dataloader_val, trainer.model, nn.CrossEntropyLoss())
    _, test_acc = compute_loss_and_accuracy(dataloader_test, trainer.model, nn.CrossEntropyLoss())

    print("\Performance of resnet18 model:")
    print(f'Training accuracy: {train_acc}')
    print(f'Training loss: {train_loss}')
    print(f'Validation accuracy: {val_acc}')
    print(f'Test accuracy: {test_acc}')