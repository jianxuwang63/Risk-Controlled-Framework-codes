from __future__ import annotations

import math


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def binomial_cdf(k: int, n: int, probability: float) -> float:
    """Numerically stable P[X <= k] for X ~ Binomial(n, probability)."""
    if n < 0 or k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if probability == 0.0:
        return 1.0
    if probability == 1.0:
        return 0.0

    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = []
    for successes in range(k + 1):
        log_comb = (
            math.lgamma(n + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(n - successes + 1)
        )
        terms.append(
            log_comb + successes * log_p + (n - successes) * log_q
        )
    return min(1.0, math.exp(_logsumexp(terms)))


def clopper_pearson_upper(
    failures: int, total: int, delta: float = 0.05
) -> float | None:
    """One-sided exact upper confidence bound for a binomial rate."""
    if total == 0:
        return None
    if not 0 <= failures <= total:
        raise ValueError("failures must be between 0 and total")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if failures == total:
        return 1.0

    low = failures / total
    high = 1.0
    for _ in range(90):
        midpoint = (low + high) / 2.0
        cdf = binomial_cdf(failures, total, midpoint)
        if cdf > delta:
            low = midpoint
        else:
            high = midpoint
    return high


def clopper_pearson_interval(
    failures: int, total: int, delta: float = 0.05
) -> tuple[float | None, float | None]:
    """Two-sided exact Clopper-Pearson interval for a binomial rate."""
    if total == 0:
        return None, None
    if not 0 <= failures <= total:
        raise ValueError("failures must be between 0 and total")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    lower = (
        0.0
        if failures == 0
        else 1.0
        - float(
            clopper_pearson_upper(
                total - failures, total, delta / 2.0
            )
        )
    )
    upper = (
        1.0
        if failures == total
        else float(clopper_pearson_upper(failures, total, delta / 2.0))
    )
    return lower, upper


def required_positives_for_zero_failures(alpha: float, delta: float) -> int:
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0, 1)")
    return math.ceil(math.log(delta) / math.log1p(-alpha))
