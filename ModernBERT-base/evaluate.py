import numpy as np


def compute_map3(eval_pred):
    logits, labels = eval_pred
    top3 = np.argsort(logits, axis=1)[:, -3:][:, ::-1]

    ap_sum = 0.0
    for i, true_label in enumerate(labels):
        hits = 0
        ap = 0.0
        for k, pred in enumerate(top3[i]):
            if pred == true_label:
                hits += 1
                ap += hits / (k + 1)
        ap_sum += ap

    return {"map3": ap_sum / len(labels)}
