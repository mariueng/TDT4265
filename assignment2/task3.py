import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # Task 3
    # Comparison between the following scenarios:
    # a) none vs improved weight init
    # b) improved weight init vs improved weight init and improved sigmoid
    # c) improved weight init vs improved weight init, improved sigmoid and momentum
    # The model with none of these improvements is trained above (model)

    # Task3a)

    # Model a: Use improved weight init
    # use_improved_weight_init = True

    # model_a = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_a = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_a, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print('Training model a ...')
    # train_history_a, val_history_a = trainer_a.train(
    #     num_epochs)
    # print('Done!')

    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history["loss"],
    #                 "Base model train", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history_a["loss"], "Model a train", npoints_to_average=10)
    # utils.plot_loss(val_history["loss"],
    #                 "Base model validation")
    # utils.plot_loss(
    #     val_history_a["loss"], "Model a validation")
    # plt.ylabel("Cross entropy loss")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.ylim([0, .4])
    # plt.subplot(1, 2, 2)
    # plt.ylim([0.90, 1.00])
    # utils.plot_loss(train_history["accuracy"], "Base model train")
    # utils.plot_loss(
    #     train_history_a["accuracy"], "Model a train")
    # utils.plot_loss(val_history["accuracy"], "Base model validation")
    # utils.plot_loss(
    #     val_history_a["accuracy"], "Model a validation")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.show()

    # # Model b: Use improved weight init and improved sigmoid
    # use_improved_sigmoid = True

    # model_b = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_b = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_b, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print('Training model b ...')
    # train_history_b, val_history_b = trainer_b.train(
    #     num_epochs)
    # print('Done!')

    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history_a["loss"],
    #                 "Model a train", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history_b["loss"], "Model b train", npoints_to_average=10)
    # utils.plot_loss(val_history_a["loss"],
    #                 "Model a validation")
    # utils.plot_loss(
    #     val_history_b["loss"], "Model b validation")
    # plt.ylabel("Cross entropy loss")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.ylim([0, .4])
    # plt.subplot(1, 2, 2)
    # plt.ylim([0.90, 1.00])
    # utils.plot_loss(train_history_a["accuracy"], "Model a train")
    # utils.plot_loss(
    #     train_history_b["accuracy"], "Model b train")
    # utils.plot_loss(val_history_a["accuracy"], "Model a validation")
    # utils.plot_loss(
    #     val_history_b["accuracy"], "Model b validation")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.show()

    # # Model c: Use improved weight init and improved sigmoid
    # use_momentum = True

    # # Change learning rate appropriately
    # learning_rate = 0.02

    # model_c = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_c = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_c, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print('Training model c ...')
    # train_history_c, val_history_c = trainer_c.train(
    #     num_epochs)
    # print('Done!')

    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history_b["loss"],
    #                 "Model b train", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history_c["loss"], "Model c train", npoints_to_average=10)
    # utils.plot_loss(val_history_b["loss"],
    #                 "Model b validation")
    # utils.plot_loss(
    #     val_history_c["loss"], "Model c validation")
    # plt.ylabel("Cross entropy loss")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.ylim([0, .4])
    # plt.subplot(1, 2, 2)
    # plt.ylim([0.90, 1.00])
    # utils.plot_loss(train_history_b["accuracy"], "Model b train")
    # utils.plot_loss(
    #     train_history_c["accuracy"], "Model c train")
    # utils.plot_loss(val_history_b["accuracy"], "Model b validation")
    # utils.plot_loss(
    #     val_history_c["accuracy"], "Model c validation")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.show()

    # Task 4a and 4b
    # Keeping all settings unchanged from Task 3
    # learning_rate = 0.02

    # use_improved_sigmoid = True
    # use_improved_weight_init = True
    # use_momentum = True

    # # 32-hidden layer unit network
    # neurons_per_layer = [32, 10]
    # model_32 = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_32 = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_32, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print("Training network with 32 hidden units...")
    # train_history_32, val_history_32 = trainer_32.train(num_epochs)
    # print("Done!")

    # # 64-hidden layer unit network
    # neurons_per_layer = [64, 10]
    # model_64 = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_64 = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_64, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print("Training network with 64 hidden units...")
    # train_history_64, val_history_64 = trainer_64.train(num_epochs)
    # print("Done!")

    # # 128-hidden layer unit network
    # neurons_per_layer = [128, 10]
    # model_128 = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_128 = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_128, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print("Training network with 128 hidden units...")
    # train_history_128, val_history_128 = trainer_128.train(num_epochs)
    # print("Done!")

    # # Plot comparisons

    # # Cross entropy loss
    # plt.figure(figsize=(20, 12))
    # utils.plot_loss(val_history_32["loss"], "Model 32")
    # utils.plot_loss(val_history_64["loss"], "Model 64")
    # utils.plot_loss(val_history_128["loss"], "Model 128")
    # plt.ylabel("Validation cross entropy loss")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.ylim([0, .4])
    # plt.show()

    # # Accuracy
    # plt.figure(figsize=(20, 12))
    # plt.ylim([0.90, 1.00])
    # utils.plot_loss(val_history_32["accuracy"], "Model 32")
    # utils.plot_loss(val_history_64["accuracy"], "Model 64")
    # utils.plot_loss(val_history_128["accuracy"], "Model 128")
    # plt.ylabel("Validation accuracy")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.show()

    # Task 4d)

    # Keep settings from earlier.
    # learning_rate = 0.02

    # use_improved_sigmoid = True
    # use_improved_weight_init = True
    # use_momentum = True

    # neurons_per_layer = [60, 60, 10]
    # model_2_hidden_layers = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_2_hidden_layers = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_2_hidden_layers, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # print("Training network with 2 hidden layers...")
    # train_history_2_hidden_layers, val_history_2_hidden_layers = trainer_2_hidden_layers.train(num_epochs)
    # print("Done!")

    # # Loss
    # plt.figure(figsize=(20, 12))
    # utils.plot_loss(train_history_2_hidden_layers["loss"], "Training", npoints_to_average=10)
    # utils.plot_loss(val_history_2_hidden_layers["loss"], "Validation")
    # plt.ylabel("Cross entropy loss")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.ylim([0, 1])
    # plt.show()

    # # Accuracy
    # plt.figure(figsize=(20, 12))
    # plt.ylim([0.90, 1.00])
    # utils.plot_loss(train_history_2_hidden_layers["accuracy"], "Training")
    # utils.plot_loss(val_history_2_hidden_layers["accuracy"], "Validation")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Number of steps")
    # plt.legend()
    # plt.show()

    # Task 4e
    learning_rate = 0.02

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model_10_hidden_layers = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_10_hidden_layers = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_10_hidden_layers, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    print("Training network with 10 hidden layers...")
    train_history_10_hidden_layers, val_history_10_hidden_layers = trainer_10_hidden_layers.train(num_epochs)
    print("Done!")

    # Loss
    plt.figure(figsize=(20, 12))
    utils.plot_loss(train_history_10_hidden_layers["loss"], "Training", npoints_to_average=10)
    utils.plot_loss(val_history_10_hidden_layers["loss"], "Validation")
    plt.ylabel("Cross entropy loss")
    plt.xlabel("Number of steps")
    plt.legend()
    plt.ylim([0, 1])
    plt.show()

    # Accuracy
    plt.figure(figsize=(20, 12))
    plt.ylim([0.90, 1.00])
    utils.plot_loss(train_history_10_hidden_layers["accuracy"], "Training")
    utils.plot_loss(val_history_10_hidden_layers["accuracy"], "Validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of steps")
    plt.legend()
    plt.show()
