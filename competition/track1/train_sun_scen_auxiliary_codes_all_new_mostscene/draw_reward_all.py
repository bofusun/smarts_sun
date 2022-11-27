import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    path_name = 'more_ratio'
    save_path = os.path.join(os.getcwd(), path_name)
    with open(os.path.join(save_path, "record_160.pkl"), "rb") as f:
        data = pickle.load(f)

    plt.subplot(3,2,1)
    time1 = range(len(data['train_return_rewards']))
    x1 = data['train_return_rewards']
    plt.plot(time1,x1,'r')
    plt.ylabel("train_rewards")
    plt.xlabel("Iteration Number")
    # plt.ylim(-100, 50)


    plt.subplot(3,2,2)
    time2 = range(len(data['test_return_rewards']))
    x2 = data['test_return_rewards']
    plt.plot(time2, x2, 'r')
    plt.ylabel("test_rewards")
    plt.xlabel("Iteration Number")
    # plt.ylim(-100, 50)


    plt.subplot(3,2,3)
    time3 = range(len(data['action_losses']))
    x3 = data['action_losses']
    plt.plot(time3, x3, 'r')
    plt.ylabel("action loss")
    plt.xlabel("Iteration Number")


    plt.subplot(3,2,4)
    time4 = range(len(data['value_losses']))
    x4 = data['value_losses']
    plt.plot(time4, x4, 'r')
    plt.ylabel("value loss")
    plt.xlabel("Iteration Number")


    plt.subplot(3,2,5)
    time5 = range(len(data['entropies']))
    x5 = data['entropies']
    plt.plot(time5, x5, 'r')
    plt.ylabel("entropies")
    plt.xlabel("Iteration Number")


    plt.subplot(3, 2, 6)
    time6 = range(len(data['resource_kl']))
    x6 = data['resource_kl']
    plt.plot(time6, x6, 'r')
    plt.ylabel("kl")
    plt.xlabel("Iteration Number")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "record_pic" + ".png"))

