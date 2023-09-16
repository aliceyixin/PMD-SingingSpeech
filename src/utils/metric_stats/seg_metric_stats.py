from charset_normalizer import detect
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from utils.metric_stats.base_metric_stats import BaseMetricStats


class SegMetricStats(BaseMetricStats):
    def __init__(self):
        super(SegMetricStats, self).__init__(metric_fn=batch_seq_pmd_scoring)
        self.saved_seqs = {}

    def append(self, ids, **kwargs):
        if self.metric_fn is None:
            raise ValueError('No metric_fn has been provided')
        self.ids.extend(ids)  # save ID
        scores = self.metric_fn(**kwargs)
        self.scores_list.extend(scores)  # save metrics
        if len(self.metric_keys) == 0:  # save metric keys
            self.metric_keys = list(self.scores_list[0].keys())

    def summarize(self, field=None):
        mean_scores = super(SegMetricStats, self).summarize()

        eps = 1e-6
        PRE = mean_scores['PRE']
        REC = mean_scores['REC']
        mean_scores['F1'] = (2 * PRE * REC) / (PRE + REC + eps)
        ER = mean_scores['ER']

        for key in mean_scores:
            mean_scores[key] = round(mean_scores[key].item(), 2)

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]


def seq_pmd_scoring(note_on_seq, note_off_seq, seg_on_seq, seg_off_seq):  # frame level scoring
    """
    Compute PMD scores of two binary sequences.
    Parameters
    ----------
    prediction : np.ndarray or torch.Tensor or list
        PMD results predicted by the model.
    target : torch.Tensor or list
        PMD ground truth.
    Returns
    -------
    pmd_scores : dict
        A dictionary of PMD scores.
    """
    TP = 0
    for i in range(len(note_on_seq)):
        note_on = note_on_seq[i]
        note_off = note_off_seq[i]

        found = False
        k = 0
        while not found and k < len(seg_on_seq):
            seg_on = seg_on_seq[k]
            seg_off = seg_off_seq[k]

            d_on = np.abs(seg_on - note_on)
            d_off = np.abs(seg_off - note_off)

            # collar 0.25 0.45
            delta = 0.2 * 48000  # 250ms
            if (d_on + d_off) < delta:
            # if d_on < delta:
                TP += 1
                break

            k += 1
        # onl = [(seg_on_seq[k]-note_on) for k in range(len(seg_on_seq))]
        # offl = [(seg_off_seq[k]-note_off) for k in range(len(seg_off_seq))]
        # d_on = np.min(np.abs(onl))
        # d_off = np.min(np.abs(offl))


    # TP = np.sum(target_onehot[:,cl] * detect_seq *prediction_onehot[:,cl])
    FP = len(seg_on_seq) -TP
    FN = len(note_on_seq) - TP
    eps = 1e-6
    PRE = TP / (TP + FP + eps) * 100
    REC = TP / (TP + FN + eps) * 100
    F1 = np.round(2 * PRE * REC / (PRE + REC + eps), 4)

    # ERROR RATE
    S = min(FN, FP)  # substitutions
    D = max(0, FN-FP)  # deletions
    I = max(0, FP-FN)  # insertions
    N = len(note_on_seq)  # num of event in the reference
    ER = np.round((S + D + I)/N, 4)


    pmd_scores = {
        'PRE': PRE,
        'REC': REC,
        'F1': F1,
        'ER': ER
    }

    return pmd_scores


def batch_seq_pmd_scoring(
        note_on_seqs=None,
        note_off_seqs=None,
        seg_on_seqs=None,
        seg_off_seqs=None,
        ):
    """
    Compute MD scores for a batch.
    Parameters
    ----------
    pred_pmd_lbl_seqs : list
        List of predicted PMD labels.
    gt_pmd_lbl_seqs : list
        List of ground truth PMD labels.


    Returns
    -------
    batch_pmd_scores : list
        list of PMD scores
    """
    
    pmd_scores = []
    for i in range(len(note_on_seqs)): # for each batch
        # PMD scores
        seq_pmd_scores = seq_pmd_scoring(note_on_seqs[i], note_off_seqs[i], seg_on_seqs[i], seg_off_seqs[i])
        
        # save scores
        pmd_scores.append(seq_pmd_scores)

    return pmd_scores
