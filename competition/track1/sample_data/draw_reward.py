import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    path_name = 'identify_model'
    save_path = os.path.join(os.getcwd(), path_name)
    with open(os.path.join(save_path, "record_100.pkl"), "rb") as f:
        data = pickle.load(f)

    plt.subplot(2,2,1)
    time1 = range(len(data['train_losses']))
    x1 = data['train_losses']
    plt.plot(time1,x1,'r')
    plt.ylabel("train_losses")
    plt.xlabel("Iteration Number")
    # plt.ylim(-100, 50)

    plt.subplot(2,2,2)
    time2 = range(len(data['test_losses']))
    x2 = data['test_losses']
    plt.plot(time2, x2, 'r')
    plt.ylabel("test_losses")
    plt.xlabel("Iteration Number")
    # plt.ylim(-100, 50)

    plt.subplot(2,2,3)
    time3 = range(len(data['train_accuracy']))
    x3 = data['train_accuracy']
    plt.plot(time3, x3, 'r')
    plt.ylabel("train_accuracy")
    plt.xlabel("Iteration Number")


    plt.subplot(2,2,4)
    time4 = range(len(data['test_accuracy']))
    x4 = data['test_accuracy']
    plt.plot(time4, x4, 'r')
    plt.ylabel("test_accuracy")
    plt.xlabel("Iteration Number")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "record_pic" + ".png"))

