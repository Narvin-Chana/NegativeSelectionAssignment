import subprocess
from turtle import filling
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, cycle
from sklearn.metrics import roc_curve, auc

n = 10
r = 3

data = {
    "syscalls/snd-cert/snd-cert.train": {
        "set1": [
            "syscalls/snd-cert/snd-cert.1.test",
            "syscalls/snd-cert/snd-cert.1.labels"
        ],
        "set2": [
            "syscalls/snd-cert/snd-cert.2.test",
            "syscalls/snd-cert/snd-cert.2.labels"
        ],
        "set3": [
            "syscalls/snd-cert/snd-cert.3.test",
            "syscalls/snd-cert/snd-cert.3.labels"
        ]
    },
    "syscalls/snd-unm/snd-unm.train": {
        "set1": [
            "syscalls/snd-unm/snd-unm.1.test",
            "syscalls/snd-unm/snd-unm.1.labels"
        ],
        "set2": [
            "syscalls/snd-unm/snd-unm.2.test",
            "syscalls/snd-unm/snd-unm.2.labels"
        ],
        "set3": [
            "syscalls/snd-unm/snd-unm.3.test",
            "syscalls/snd-unm/snd-unm.3.labels"
        ]
    }
}

alpha_sets = {"syscalls/snd-cert/snd-cert.train": "syscalls/snd-cert/snd-cert.alpha", "syscalls/snd-unm/snd-unm.train": "syscalls/snd-unm/snd-unm.alpha"}

# Iterate over the training sets
for train_set in data:

    # Iterate over the test sets
    for set in data[train_set]:
        test_set = data[train_set][f'{set}'][0]

        # Runs negsel2.jar
        subprocess.run(f'java -jar negsel2.jar -self {train_set} -n {n} -r {r} -c -l -k -alphabet file://{alpha_sets[train_set]} < {test_set} > {test_set}result', shell=True)

        # Get expected results (labels)
        y_true = np.genfromtxt(data[train_set][f'{set}'][1])

        y_score = np.zeros(len(y_true))

        # Get actual results (what negsel2.jar returns)     
        # Gets the average of each line by skipping all null elements in the line.
        with open(f'{test_set}result', 'r') as file:
            lines = file.readlines()

            line_counter = 0

            for line in lines:
                splitLine = line.split()
                
                lineVal = 0
                numElemsInLine = 0

                for elem in splitLine:
                    if float(elem) != 0 and elem != 'NaN':
                        lineVal += float(elem)
                        numElemsInLine +=1
                
                y_score[line_counter] = lineVal / numElemsInLine if numElemsInLine != 0 else 0
                line_counter += 1

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label=f"ROC curve for dataset {test_set} (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver operating characteristic curve with n={n} and r={r}")
        plt.legend(loc="lower right")
        plt.show()

        print(f"Set {test_set} done.")
    print(f"Set {train_set} done.")