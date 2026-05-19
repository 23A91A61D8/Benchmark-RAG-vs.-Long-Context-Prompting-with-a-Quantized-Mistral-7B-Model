from rouge_score import rouge_scorer
from bert_score import score


def calculate_rouge(reference, prediction):

    rouge = rouge_scorer.RougeScorer(
        ['rouge1', 'rougeL'],
        use_stemmer=True
    )

    scores = rouge.score(
        reference,
        prediction
    )

    return scores


def calculate_bertscore(predictions, references):

    P, R, F1 = score(
        predictions,
        references,
        lang="en"
    )

    return F1.mean().item()