import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """

    # Make predictions
    y_pred = model.forward(X)

    # Calculate accuracy
    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(targets, axis=1)) / y_pred.shape[0]
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """

        # Perform forward step
        y_pred = self.model.forward(X_batch)

        # Perform backward step
        self.model.backward(X_batch, y_pred, Y_batch)

        # Perform gradient descent
        self.model.w = self.model.w - self.learning_rate * self.model.grad

        # Calculate loss
        loss = cross_entropy_loss(Y_batch, y_pred)

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)
    
    # Store weights before training with L2 regularization
    n = 28

    # Weights for lambda = 0.0
    weights = np.zeros((n * 2, n * trainer.model.num_outputs))
    for i in range(trainer.model.num_outputs):
        weights[:n, (n * i):(n * (i + 1))] = np.reshape(trainer.model.w[:-1, i], (n, n))

    model1 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.


    # Weights for lambda = 2.0
    for i in range(trainer.model.num_outputs):
        weights[n:, (n * i):(n * (i + 1))] = np.reshape(trainer.model.w[:-1, i], (n, n))

    # Plotting of softmax weights (Task 4b)
    # plt.imsave("task4b_softmax_weight.png", weights, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [2, .2, .02, .002]
    weights = np.zeros(len(l2_lambdas))
    training_steps_counter = 0
    for lambda_constant in l2_lambdas:

        # Instanstiate Model for each lambda constant
        model = SoftmaxModel(l2_reg_lambda=lambda_constant)

        # Instantiate Trainer
        trainer = SoftmaxTrainer(model, learning_rate, batch_size, shuffle_dataset, X_train, Y_train, X_val, Y_val)

        # Store training and validation history
        train_history, val_history = trainer.train(num_epochs)

        # Continously plot accuracy together with lambda constant
        utils.plot_loss(val_history["accuracy"], fr'$\lambda=${lambda_constant}')

        # Normalize weights and store them for each step
        weights[training_steps_counter] = np.linalg.norm(trainer.model.w)

        # Increment training steps counter
        training_steps_counter += 1

    # Task 4c) - Plotting the accuracy of the different models
    plt.figure(1)
    plt.xlabel("Training steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")

    # Task 4d - Plotting of the l2 norm for each weight
    plt.figure(2)
    print(l2_lambdas)
    print(weights)
    plt.plot(l2_lambdas, weights)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Length")
    plt.savefig("task4d_l2_reg_norms.png")
