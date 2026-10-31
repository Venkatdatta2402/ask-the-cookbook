"""Retrieval metrics against a graded golden set: Precision@K, Recall@K, MRR and NDCG@K.

A passage is (file, (first line, last line)). The golden set lists, for a question, the passages that matter and a
grade for each (3 states the answer, 2 a needed part of it, 1 helps a little); every passage not listed is grade 0.

Matching comes first, and everything else is arithmetic on what it returns. A retrieved passage takes the grade of
the golden passage it matches: same file, and line ranges overlapping by at least `min_jaccard` (shared lines over
all lines). Exact spans would do on the index the golden set was graded on; the overlap keeps a refreshed index,
whose chunk edges have moved a line or two, measurable. Each golden passage is credited once -- two retrieved
passages cut from one golden one do not both score -- and each retrieved passage credits at most one.

    gains       the grade each retrieved passage earned, in rank order: [3, 0, 1, 0, 2]
    relevant    grade 1 and up: Precision@K and MRR count a passage that helps at all
    needed      grade 2 and up: Recall@K counts what an answer cannot do without. Over every relevant passage a
                question with 35 of them could never pass 5/35 in five places, and the number would say nothing.
                Even over needed ones, k places cap it: `best_recall_at_k` is that cap, reported beside it
    NDCG@K      every grade, as gain 2**grade - 1, against the best ordering the golden set allows
"""

import math


def jaccard(a, b):
    """Shared lines over all lines, for two inclusive line ranges."""
    shared = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if shared <= 0:
        return 0.0
    return shared / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - shared)


def gains(retrieved, golden, min_jaccard=0.5):
    """The grade each retrieved passage earned, in rank order (0 for none).

    `retrieved`: [(file, (a, b))], best first. `golden`: [(file, (a, b), grade)]. Each golden passage goes to the
    first retrieved passage that matches it; where one retrieved passage matches several, the best overlap wins,
    then the higher grade."""
    taken, out = set(), []
    for file, lines in retrieved:
        best = None
        for i, (g_file, g_lines, grade) in enumerate(golden):
            if i in taken or g_file != file:
                continue
            overlap = jaccard(lines, g_lines)
            if overlap >= min_jaccard and (best is None or (overlap, grade) > best[:2]):
                best = (overlap, grade, i)
        if best is None:
            out.append(0)
        else:
            taken.add(best[2])
            out.append(best[1])
    return out


def precision_at_k(gains_, k, min_grade=1):
    """Of the first k places, the share holding a relevant passage. A place left empty -- fewer than k returned, or
    none because the floor refused the question -- counts as not relevant."""
    return sum(g >= min_grade for g in gains_[:k]) / k


def recall_at_k(gains_, grades, k, min_grade=2):
    """Of the golden passages graded `min_grade` or above, the share found in the first k places. None when the
    golden set has none."""
    wanted = sum(g >= min_grade for g in grades)
    if not wanted:
        return None
    return sum(g >= min_grade for g in gains_[:k]) / wanted


def best_recall_at_k(grades, k, min_grade=2):
    """The most Recall@K a perfect search could reach: k places hold at most k of the needed passages, so a question
    with 16 of them tops out at 5/16. Read beside recall, so that a capped number is not taken for a poor one."""
    wanted = sum(g >= min_grade for g in grades)
    return min(k, wanted) / wanted if wanted else None


def reciprocal_rank(gains_, min_grade=1):
    """1 / the place of the first relevant passage; 0 when there is none."""
    return next((1 / place for place, g in enumerate(gains_, 1) if g >= min_grade), 0.0)


def dcg(grades, k):
    return sum((2 ** g - 1) / math.log2(place + 1) for place, g in enumerate(grades[:k], 1))


def ndcg_at_k(gains_, grades, k):
    """DCG of the first k places over the DCG of the golden passages in their best order. 0 when nothing is graded."""
    best = dcg(sorted(grades, reverse=True), k)
    return dcg(gains_, k) / best if best else 0.0


def score(retrieved, golden, k=5, min_jaccard=0.5):
    """Every metric for one question, and the gains they were computed from."""
    earned = gains(retrieved, golden, min_jaccard)
    grades = [g for _, _, g in golden]
    return {"gains": earned,
            "precision": precision_at_k(earned, k),
            "recall": recall_at_k(earned, grades, k),
            "recall_best": best_recall_at_k(grades, k),
            "mrr": reciprocal_rank(earned),
            "ndcg": ndcg_at_k(earned, grades, k)}


def mean(rows, metric):
    """The mean of a metric over the questions that have one (recall is None for a question with no needed passage)."""
    values = [r[metric] for r in rows if r[metric] is not None]
    return sum(values) / len(values) if values else None
