""""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

"""
Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
"""

from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    all_ngrams = []

    #only iterate over valid indices
    for i in range(len(seq) - n + 1):
        all_ngrams.append(seq[i:i + n])

    return all_ngrams


def n_gram_precision(
    reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    reference_ngrams = grouper(reference, n)
    candidate_ngrams = grouper(candidate, n)

    matches = 0
    for ngram in candidate_ngrams:
        if ngram in reference_ngrams:
            matches += 1

    # Calculate precision for n-grams
    if len(candidate_ngrams) == 0:
        p_n = 0.0
    else:
        p_n = matches / len(candidate_ngrams)

    return p_n


def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    if len(candidate) == 0:
        BP = 0.0
    elif len(candidate) > len(reference):
        BP = 1.0
    else:
        BP = 1.0 - (len(reference) / len(candidate))
        BP = exp(BP)

    return BP


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    """
    Calculate the BLEU score.  Please scale the BLEU score by 100.0

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """
    prec = []
    for i in range(1, n + 1):
        prec.append(n_gram_precision(reference, candidate, i))

    BP = brevity_penalty(reference, candidate)

    bleu = 1
    for p in prec:
        bleu *= p

    # print(bleu)
    # print(BP)
    
    bleu = bleu ** (1/n)
    bleu = BP * bleu
    #Scale since it ranges from 0 to 1
    return bleu * 100
