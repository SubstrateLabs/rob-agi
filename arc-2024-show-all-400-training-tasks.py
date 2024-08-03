import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import json
import os
from pathlib import Path
from glob import glob
from subprocess import Popen, PIPE, STDOUT


base_path = "./data/inputs/"


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


training_challenges = load_json(base_path + "arc-agi_training_challenges.json")
training_solutions = load_json(base_path + "arc-agi_training_solutions.json")
evaluation_challenges = load_json(base_path + "arc-agi_evaluation_challenges.json")
evaluation_solutions = load_json(base_path + "arc-agi_evaluation_solutions.json")
test_challenges = load_json(base_path + "arc-agi_test_challenges.json")

cmap = colors.ListedColormap(
    ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
)
norm = colors.Normalize(vmin=0, vmax=9)

plt.figure(figsize=(3, 1), dpi=150)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()


def plot_task(task, task_solutions, i, t):
    """Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app"""

    num_train = len(task["train"])
    num_test = len(task["test"])

    w = num_train + num_test
    fig, axs = plt.subplots(2, w, figsize=(3 * w, 3 * 2))
    plt.suptitle(f"Set #{i}, {t}:", fontsize=20, fontweight="bold", y=1)
    # plt.subplots_adjust(hspace = 0.15)
    # plt.subplots_adjust(wspace=20, hspace=20)

    for j in range(num_train):
        plot_one(axs[0, j], j, "train", "input")
        plot_one(axs[1, j], j, "train", "output")

    plot_one(axs[0, j + 1], 0, "test", "input")

    answer = task_solutions
    input_matrix = answer

    axs[1, j + 1].imshow(input_matrix, cmap=cmap, norm=norm)
    axs[1, j + 1].grid(True, which="both", color="lightgrey", linewidth=0.5)
    axs[1, j + 1].set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    axs[1, j + 1].set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    axs[1, j + 1].set_xticklabels([])
    axs[1, j + 1].set_yticklabels([])
    axs[1, j + 1].set_title("Test output")

    axs[1, j + 1] = plt.figure(1).add_subplot(111)
    axs[1, j + 1].set_xlim([0, num_train + 1])

    for m in range(1, num_train):
        axs[1, j + 1].plot([m, m], [0, 1], "--", linewidth=1, color="black")

    axs[1, j + 1].plot([num_train, num_train], [0, 1], "-", linewidth=3, color="black")

    axs[1, j + 1].axis("off")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor("black")
    fig.patch.set_facecolor("#dddddd")
    plt.tight_layout()

    print(f"#{i}, {t}")
    # plt.show()
    # save to a file:
    plt.savefig(f"./data/task_images/{t}.png")
    print()
    print()


def plot_one(ax, i, train_or_test, input_or_output):
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)

    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    ax.set_title(train_or_test + " " + input_or_output)


chunks = [list(training_challenges)[i : i + 50] for i in range(0, 400, 50)]

only_chunk = sys.argv[1]
if only_chunk:
    chunks = [chunks[int(only_chunk)]]
    print(f"Processing only chunk {only_chunk}")

for chunk in chunks:
    for i, t in enumerate(chunk):
        task = training_challenges[t]
        task_solution = training_solutions[t][0]
        plot_task(task, task_solution, i, t)
#
# for i in range(0, 50):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
# for i in range(50, 100):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
#
# for i in range(100, 150):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
#
# for i in range(150, 200):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
# for i in range(200, 250):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
# for i in range(250, 300):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
# for i in range(300, 350):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
#
# for i in range(350, 400):
#     t = list(training_challenges)[i]
#     task = training_challenges[t]
#     task_solution = training_solutions[t][0]
#     plot_task(task, task_solution, i, t)
