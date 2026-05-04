"""Compute Gwet's AC1 (and AC2 for ordinal anxiety) on the existing
reliability + self-retest pairs. AC1 corrects for the prevalence/marginal-
imbalance pathology in Cohen's kappa.

Loads:
  results/judge_reliability.json    (run-1 vs GP)
  results/judge_self_retest.json    (run-1 vs run-2 paired predictions)

Reports AC1 for binary `over_cautious` and AC2 (linear/quadratic weights) for
the 0-10 anxiety index, with 95% CIs from the standard Gwet variance formula.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAG = "The AI exaggerates the risk or suggests actions disproportionate to the scenario."


def gwet_ac1_binary(r1, r2):
    """Gwet's AC1 for two binary raters. r1, r2 are equal-length 0/1 lists.

    Returns (AC1, SE, lo, hi) — point estimate plus 95% CI from item-level variance.
    Reference: Gwet 2008, "Computing inter-rater reliability and its variance
    in the presence of high agreement", Br J Math Stat Psychol 61:29-48.
    """
    assert len(r1) == len(r2) and len(r1) > 0
    n = len(r1)
    # Per-item agreement indicator
    a = [1 if r1[i] == r2[i] else 0 for i in range(n)]
    pa = sum(a) / n
    # Marginal probabilities of category 1 averaged across raters
    p1_avg = (sum(r1) + sum(r2)) / (2 * n)
    # Pe under Gwet's chance-correction (q=2 categories): 2*pi*(1-pi)
    pe = 2 * p1_avg * (1 - p1_avg)
    ac1 = (pa - pe) / (1 - pe) if pe < 1 else 0.0

    # Item-level AC1 contribution for variance estimate
    ac1_i = []
    for i in range(n):
        agree_i = a[i]
        # Item-level pi: avg of the two raters' indicator at item i
        pi_i = (r1[i] + r2[i]) / 2
        pe_i = 2 * pi_i * (1 - pi_i)
        gamma_i = (agree_i - pe_i) / (1 - pe) if pe < 1 else 0.0
        ac1_i.append(gamma_i)
    mean_g = sum(ac1_i) / n
    var = sum((g - mean_g) ** 2 for g in ac1_i) / (n * (n - 1)) if n > 1 else 0.0
    se = math.sqrt(var)
    lo, hi = ac1 - 1.96 * se, ac1 + 1.96 * se
    return ac1, se, lo, hi


def gwet_ac2_ordinal(r1, r2, K, weight="quadratic"):
    """Gwet's AC2 for two raters on an ordinal K-point scale (categories 0..K-1).

    weight: "linear" or "quadratic" disagreement weights.
    Returns (AC2, SE, lo, hi).
    """
    assert len(r1) == len(r2)
    n = len(r1)
    # Weight matrix w[a][b] in [0,1]; 1 = full agreement, 0 = max disagreement
    w = [[0.0] * K for _ in range(K)]
    max_d = (K - 1) ** 2 if weight == "quadratic" else (K - 1)
    for a in range(K):
        for b in range(K):
            d = (a - b) ** 2 if weight == "quadratic" else abs(a - b)
            w[a][b] = 1.0 - d / max_d

    pa_i = [w[r1[i]][r2[i]] for i in range(n)]
    pa = sum(pa_i) / n

    # Marginal pi_k = avg rater frequency for category k
    pi = [0.0] * K
    for x in r1:
        pi[x] += 1
    for x in r2:
        pi[x] += 1
    pi = [p / (2 * n) for p in pi]

    # Tw = sum over (a,b) of w[a][b]; for K cats with quad/linear weights:
    Tw = sum(w[a][b] for a in range(K) for b in range(K))
    # Gwet AC2 chance: pe = (Tw / (K*(K-1))) * sum_k pi_k*(1-pi_k)
    pe_factor = sum(p * (1 - p) for p in pi)
    pe = (Tw / (K * (K - 1))) * pe_factor if K > 1 else 0.0
    ac2 = (pa - pe) / (1 - pe) if pe < 1 else 0.0

    # Item-level for variance
    ac2_i = []
    for i in range(n):
        # Item-specific pe_i: contribution at this item
        # Use the same Gwet form with item-specific marginal indicators.
        pi_item = [0.0] * K
        pi_item[r1[i]] += 0.5
        pi_item[r2[i]] += 0.5
        pe_item_factor = sum(p * (1 - p) for p in pi_item)
        pe_i = (Tw / (K * (K - 1))) * pe_item_factor if K > 1 else 0.0
        gamma_i = (pa_i[i] - pe_i) / (1 - pe) if pe < 1 else 0.0
        ac2_i.append(gamma_i)
    mean_g = sum(ac2_i) / n
    var = sum((g - mean_g) ** 2 for g in ac2_i) / (n * (n - 1)) if n > 1 else 0.0
    se = math.sqrt(var)
    return ac2, se, ac2 - 1.96 * se, ac2 + 1.96 * se


def main():
    rel = json.load((ROOT / "results" / "judge_reliability.json").open())
    rt = json.load((ROOT / "results" / "judge_self_retest.json").open())

    # vs GP — use run-1 predictions
    valid = [r for r in rel["predictions"] if "v4pro_over_cautious" in r]
    gp_bin = [1 if r["gp_label"] == EXAG else 0 for r in valid]
    judge_bin = [1 if r["v4pro_over_cautious"] else 0 for r in valid]
    print(f"=== vs GP (n={len(valid)}, prevalence GP=exag: {sum(gp_bin)/len(gp_bin):.3f}) ===")
    ac1, se, lo, hi = gwet_ac1_binary(gp_bin, judge_bin)
    print(f"  Gwet's AC1 (over_cautious): {ac1:.4f}  (95% CI {lo:.4f}, {hi:.4f}; SE {se:.4f})")
    raw_agreement = sum(1 for a, b in zip(gp_bin, judge_bin) if a == b) / len(gp_bin)
    print(f"  raw agreement: {raw_agreement:.4f}")

    # Self-agreement — use paired run1/run2
    paired = rt["paired"]
    r1_bin = [1 if p["v4pro_run1_oc"] else 0 for p in paired]
    r2_bin = [1 if p["v4pro_run2_oc"] else 0 for p in paired]
    print(f"\n=== Self-retest (n={len(paired)}) ===")
    ac1, se, lo, hi = gwet_ac1_binary(r1_bin, r2_bin)
    print(f"  Gwet's AC1 (over_cautious): {ac1:.4f}  (95% CI {lo:.4f}, {hi:.4f}; SE {se:.4f})")

    # Anxiety index: 0-10 ordinal, 11 categories
    a1 = [p["v4pro_run1_anx"] for p in paired]
    a2 = [p["v4pro_run2_anx"] for p in paired]
    K = 11
    ac2_q, se_q, lo_q, hi_q = gwet_ac2_ordinal(a1, a2, K, weight="quadratic")
    ac2_l, se_l, lo_l, hi_l = gwet_ac2_ordinal(a1, a2, K, weight="linear")
    print(f"  Gwet's AC2 (anxiety, quadratic weights): {ac2_q:.4f}  (95% CI {lo_q:.4f}, {hi_q:.4f})")
    print(f"  Gwet's AC2 (anxiety, linear weights):    {ac2_l:.4f}  (95% CI {lo_l:.4f}, {hi_l:.4f})")

    # Save
    out = {
        "vs_gp": {
            "n": len(valid),
            "gp_prevalence_exaggerates": sum(gp_bin) / len(gp_bin),
            "ac1_over_cautious": gwet_ac1_binary(gp_bin, judge_bin),
        },
        "self_retest": {
            "n": len(paired),
            "ac1_over_cautious": gwet_ac1_binary(r1_bin, r2_bin),
            "ac2_anxiety_quadratic": gwet_ac2_ordinal(a1, a2, K, "quadratic"),
            "ac2_anxiety_linear": gwet_ac2_ordinal(a1, a2, K, "linear"),
        },
    }
    out_path = ROOT / "results" / "judge_ac1.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
