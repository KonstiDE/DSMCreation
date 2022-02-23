import os
import csv
import matplotlib.pyplot as plt


def plot(path):
    metrics = {}

    training_loss = []
    validation_loss = []

    training_mse = []
    validation_mse = []
    training_mae = []
    validation_mae = []

    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        index = 0
        for row in spamreader:
            s = ', '.join(row)

            array = s.split(", ")

            if '#' in s:
                for type in array:
                    metrics['' + type] = None
            else:
                for i in range(len(array)):
                    array[i] = float(array[i])
                metrics['' + list(metrics)[index - 1]] = array

            index += 1

        for type in metrics.keys():
            if type == "# tloss":
                training_loss = metrics[type]
            elif type == "vloss":
                validation_loss = metrics[type]
            elif type == "tmae":
                training_mae = metrics[type]
            elif type == "tmse":
                training_mse = metrics[type]
            elif type == "vmae":
                validation_mae = metrics[type]
            elif type == "vmse":
                validation_mse = metrics[type]

        plt.figure()
        plt.plot(training_loss, 'b', label="Training loss")
        plt.plot(validation_loss, 'r', label="Validation loss")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.savefig(
            "loss.png"
        )
        plt.show()

        plt.figure()
        plt.plot(training_mae, 'b', label="Training MAE")
        plt.plot(validation_mae, 'orange', label="Validation MAE")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.savefig(
            "mae.png"
        )
        plt.show()

        plt.figure()
        plt.plot(training_mse, 'b', label="Training MSE")
        plt.plot(validation_mse, 'orange', label="Validation MSE")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.savefig(
            "mse.png"
        )
        plt.show()


if __name__ == '__main__':
    plot("/home/fkt48uj/nrw/results_RMSProp_Adam/metrics.csv")
