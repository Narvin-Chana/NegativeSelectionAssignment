import subprocess
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

n = 10
r = 4

files = {
    "tagalog.test",
    "lang/hiligaynon.txt",
    "lang/middle-english.txt",
    "lang/plautdietsch.txt",
    "lang/xhosa.txt"
}

subprocess.run(f'java -jar negsel2.jar -self english.train -n {n} -r {r} -c -l < ./english.test > res.txt', shell=True)
englishData = np.loadtxt('res.txt')

for test_set in files:

    subprocess.run(f'java -jar negsel2.jar -self english.train -n {n} -r {r} -c -l < {test_set} > ./{test_set}result', shell=True)

    test_data = np.loadtxt(f'{test_set}result')

    y_score = np.concatenate((englishData, test_data))
    y_true = np.array(([1 for _ in range(len(englishData))] + [0 for _ in range(len(test_data))]))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=0)
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