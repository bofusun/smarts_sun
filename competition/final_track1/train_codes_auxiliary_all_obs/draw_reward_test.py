import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    path_name = 'aux_01_origin_reward_step_0001'
    save_path = os.path.join(os.getcwd(), path_name)
    with open(os.path.join(save_path, "record_310.pkl"), "rb") as f:
        data = pickle.load(f)

    plt.subplot(3,2,1)
    time1 = range(len(data['test_reward_scenario_0']))
    x1 = data['test_reward_scenario_0']
    plt.plot(time1,x1,'r')
    plt.ylabel("scenario_0")
    plt.xlabel("Iteration Number")
    # plt.ylim(-100, 50)

    plt.subplot(3,2,2)
    time2 = range(len(data['test_reward_scenario_1']))
    x2 = data['test_reward_scenario_1']
    plt.plot(time2, x2, 'r')
    plt.ylabel("scenario_1")
    plt.xlabel("Iteration Number")
    # plt.ylim(-100, 50)

    plt.subplot(3,2,3)
    time3 = range(len(data['test_reward_scenario_2']))
    x3 = data['test_reward_scenario_2']
    plt.plot(time3, x3, 'r')
    plt.ylabel("scenario_2")
    plt.xlabel("Iteration Number")


    plt.subplot(3,2,4)
    time4 = range(len(data['test_reward_scenario_3']))
    x4 = data['test_reward_scenario_3']
    plt.plot(time4, x4, 'r')
    plt.ylabel("scenario_3")
    plt.xlabel("Iteration Number")


    plt.subplot(3,2,5)
    time5 = range(len(data['test_reward_scenario_4']))
    x5 = data['test_reward_scenario_4']
    plt.plot(time5, x5, 'r')
    plt.ylabel("scenario_4")
    plt.xlabel("Iteration Number")


    plt.subplot(3, 2, 6)
    time6 = range(len(data['test_reward_scenario_5']))
    x6 = data['test_reward_scenario_5']
    plt.plot(time6, x6, 'r')
    plt.ylabel("scenario_5")
    plt.xlabel("Iteration Number")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "record_test_reward_pic" + ".png"))

