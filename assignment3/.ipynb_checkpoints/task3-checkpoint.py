import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ImprovedModelOne(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # ------- Layer 1 -------
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # ------- Layer 2 -------
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # ------- Layer 3 -------
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # ------- Flatten -------
            nn.Flatten()
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16] 
        self.num_output_features = 4 * 4 * 128

        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        feature_extractor = self.feature_extractor(x)
        out = self.classifier(feature_extractor)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class ImprovedModelTwo(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # ------- Layer 1 -------
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # ------- Layer 2 -------
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # ------- Layer 3 -------
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # ------- Flatten -------
            nn.Flatten()
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16] 
        self.num_output_features = 4 * 4 * 128

        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        feature_extractor = self.feature_extractor(x)
        out = self.classifier(feature_extractor)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    
    # First model

    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    m1 = ImprovedModelOne(image_channels=3, num_classes=10)
    trainer_m1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        m1,
        dataloaders
    )
    print("Training first model ...")
    trainer_m1.train()
    print("Training done!")
    create_plots(trainer_m1, "task3_model_one")

    # Print results from model

    dl_train_m1, dl_val_m1, dl_test_m1 = dataloaders

    train_loss_m1, train_acc_m1 = compute_loss_and_accuracy(dl_train_m1, trainer_m1.model, nn.CrossEntropyLoss())
    _, val_acc_m1 = compute_loss_and_accuracy(dl_val_m1, trainer_m1.model, nn.CrossEntropyLoss())
    _, test_acc_m1 = compute_loss_and_accuracy(dl_test_m1, trainer_m1.model, nn.CrossEntropyLoss())

    print('Accuracies on different datasets for model one: ')
    print(f'Training accuracy: {train_acc_m1}')
    print(f'Training loss: {train_loss_m1}')
    print(f'Training accuracy: {val_acc_m1}')
    print(f'Training accuracy: {test_acc_m1}')

    # Second model

    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    m2 = ImprovedModelTwo(image_channels=3, num_classes=10)
    trainer_m2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        m2,
        dataloaders
    )
    print("Training second model ...")
    trainer_m2.train()
    print("Training done!")
    create_plots(trainer_m2, "task3_model_two")

    dl_train_m2, dl_val_m2, dl_test_m2 = dataloaders

    train_loss_m2, train_acc_m2 = compute_loss_and_accuracy(dl_train_m2, trainer_m2.model, nn.CrossEntropyLoss())
    _, val_acc_m2 = compute_loss_and_accuracy(dl_val_m2, trainer_m2.model, nn.CrossEntropyLoss())
    _, test_acc_m2 = compute_loss_and_accuracy(dl_test_m2, trainer_m2.model, nn.CrossEntropyLoss())
    
    print('Accuracies on different datasets for model one: ')
    print(f'Training accuracy: {train_acc_m2}')
    print(f'Training loss: {train_loss_m2}')
    print(f'Training accuracy: {val_acc_m2}')
    print(f'Training accuracy: {test_acc_m2}')

if __name__ == "__main__":
    main()