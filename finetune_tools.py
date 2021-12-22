import matplotlib.pyplot as plt

def visualize_history(history):
    # epoch = trained_models.epoch # list


    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize trained_models for loss
    # plt.plot(trained_models.trained_models['loss'])
    # plt.plot(trained_models.trained_models['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()