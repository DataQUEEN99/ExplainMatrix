
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_to_index = {l:i for i,l in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t,p in zip(y_true, y_pred):
        cm[label_to_index[t]][label_to_index[p]] += 1

    return cm, labels


def metrics(cm):
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    accuracy = np.sum(tp) / np.sum(cm)
    precision = np.mean(tp / (tp + fp + 1e-9))
    recall = np.mean(tp / (tp + fn + 1e-9))
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def insights(cm, labels):
    insights = []

    for i, label in enumerate(labels):
        tp = cm[i,i]
        fn = np.sum(cm[i,:]) - tp
        fp = np.sum(cm[:,i]) - tp

        if fn > fp:
            insights.append(f"{label}: model missing true cases (low recall)")
        elif fp > fn:
            insights.append(f"{label}: too many false positives")

    if np.trace(cm)/np.sum(cm) < 0.7:
        insights.append("Overall model performance is weak")

    return insights


def plot(cm, labels, normalize=False):
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, round(cm[i,j],2), ha='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def report(y_true, y_pred):
    cm, labels = confusion_matrix(y_true, y_pred)
    m = metrics(cm)
    ins = insights(cm, labels)

    print("Labels:", labels)
    print("\nConfusion Matrix:\n", cm)

    print("\nMetrics:")
    for k,v in m.items():
        print(f"{k}: {round(v,3)}")

    print("\nInsights:")
    for i in ins:
        print("-", i)

    return cm
