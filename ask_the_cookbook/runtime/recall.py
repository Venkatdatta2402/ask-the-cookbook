"""Chapter 4's min-max normalisation: the one part of its ranking (`Recall`) that chapter 5's selection uses."""


def unit(values):
    """Min-max to [0, 1]. All-equal collapses to 0.5 rather than dividing by zero."""
    lo, hi = min(values), max(values)
    return [0.5 if hi == lo else (v - lo) / (hi - lo) for v in values]
