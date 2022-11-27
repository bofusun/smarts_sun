import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    path_name = "/root/deeplearningnew/SMARTS4/smarts/scenarios/merge/3lane_single_agent/missions.pkl"
    with open(path_name, "rb") as f:
        data = pickle.load(f)
    print(data)