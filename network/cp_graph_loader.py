import os
import torch
import statistics as s

import matplotlib.pyplot as plt


def load_graphs_from_checkpoint(model_path, epoch):
    if os.path.isfile(model_path + "model_epoch" + str(epoch) + ".pt"):
        checkpoint = torch.load(model_path + "model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_mae = checkpoint['training_maes']
        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']
        overall_training_zncc = checkpoint['training_znccs']
        overall_validation_mae = checkpoint['validation_maes']
        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']
        overall_validation_zncc = checkpoint['validation_znccs']

        plt.figure()
        plt.plot(overall_training_loss, 'b', label="Training loss")
        plt.plot(overall_validation_loss, 'r', label="Validation loss")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_mae, 'b', label="Training MAE")
        plt.plot(overall_validation_mae, 'orange', label="Validation MAE")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_mse, 'b', label="Training MSE")
        plt.plot(overall_validation_mse, 'orange', label="Validation MSE")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_ssim, 'b', label="Training SSIM")
        plt.plot(overall_validation_ssim, 'orange', label="Validation SSIM")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_zncc, 'b', label="Training ZNCC")
        plt.plot(overall_validation_zncc, 'orange', label="Validation ZNCC")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        print('MAE train: ' + str(s.mean(overall_training_mae)))
        print('MAE valid: ' + str(s.mean(overall_validation_mae)))

        print('MSE train: ' + str(s.mean(overall_training_mse)))
        print('MSE valid: ' + str(s.mean(overall_validation_mse)))

        print('SSIM train: ' + str(s.mean(overall_training_ssim)))
        print('SSIM valid: ' + str(s.mean(overall_validation_ssim)))

        print('ZNCC train: ' + str(s.mean(overall_training_zncc)))
        print('ZNCC valid: ' + str(s.mean(overall_validation_zncc)))

    else:
        print("No model found within {} and epoch {}".format(
            model_path,
            str(epoch)
        ))


if __name__ == '__main__':
    load_graphs_from_checkpoint("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/", 23)
