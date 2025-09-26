# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats


def evaluate_summary(predicted_summary, user_summary, score, gtscore, eval_method='avg'):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray gt_summary: The user defined ground truth summaries (or summary).
    """

    # evaluation method for summe and tvsum dataset also applicable on yt8m
    max_len = max(len(predicted_summary), user_summary.shape[0])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    f_scores = []
    for user in range(1):  # user_summary.shape[0]
        G[:user_summary.shape[0]] = user_summary
        overlapped = S & G
        # Compute precision, recall, f-score
        precision = sum(overlapped) / sum(S + 1e-8)
        recall = sum(overlapped) / sum(G + 1e-8)
        # print("max_len", max_len)
        # print("sum of overlapped", sum(overlapped))
        # print("sum of S", sum(S))
        # print("sum of G", sum(G))
        # print("shape of S", np.array(S).shape)
        # print("shape of G", np.array(G).shape)
        # print("precision", precision)
        # print("recall", recall)

        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_scores.append((2 * precision * recall * 100) / (precision + recall))

    if eval_method == 'max':
        f_score_result = max(f_scores)
    else:
        f_score_result = sum(f_scores) / len(f_scores)

    y_pred2 = score.numpy()
    y_true2 = gtscore.numpy()
    pS = stats.spearmanr(y_pred2, y_true2)[0]
    kT = stats.kendalltau(stats.rankdata(np.array(y_pred2)), stats.rankdata(np.array(y_true2)))[0]
    return f_score_result, kT, pS