import string
import re
import json
import sys
import os
import logging
from collections import Counter, defaultdict
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def compute_txt_metrics(predictions, references, metrics, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    metric_values = defaultdict(int)
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        if "exact_match" in metrics:
            metric_values["exact_match"] += metric_max_over_ground_truths(
                exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )

        if "partial_match" in metrics:
            metric_values["partial_match"] += metric_max_over_ground_truths(
                partial_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            
        if "rouge1" in metrics:
            metric_values["rouge1"] += metric_max_over_ground_truths(
                rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )

        if "rougeL" in metrics:
            metric_values["rougeL"] += metric_max_over_ground_truths(
                rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )

    metric_values = {k: v/len(references) for k, v in metric_values.items()}
    return metric_values


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def partial_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(ground_truth) in normalize_answer(prediction))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_grouped_metrics(preds, labels, groups, compute_metrics, metrics):
    assert len(preds) == len(labels) == len(groups)

    examples_by_group = {}
    for pred, label, group in zip(preds, labels, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, label))

    results = {}
    for group, group_examples in examples_by_group.items():
        preds, labels = zip(*group_examples)
        group_metrics = compute_metrics(predictions=preds, references=labels, metrics=metrics)
        for key, value in group_metrics.items():
            results[f"{key}_{group}"] = value
    return results
