# Pre-registered predictions

Written **before** running the sweep. Scored honestly afterwards, including the misses.
Committing this first is the point: it is what stops the study from becoming a search for
a flattering result.

| # | Prediction | Rationale | Outcome |
|---|---|---|---|
| P1 | Accuracy rises steeply from k=1 to k=8 and is flat from k=16 to k=32 | Short videos saturate fast | |
| P2 | `mmr` beats `uniform` by 1–3 pts at k=2–4, and is n.s. by k=16 | Matches FOCUS's reported +0.7–2.1 on short-video benchmarks | |
| P3 | `topk_sim` underperforms `mmr` at k>=4 | Top-k collapses onto one moment, loses temporal coverage | |
| P4 | Selection helps descriptive/temporal-localization types more than causal types | Causal questions need the whole event chain, not one frame | |
| P5 | >=55% of questions answered correctly at k=4 are still correct at k=16 | Most questions are easy; fixed budgets overpay for them | |
| P6 | `margin` beats `maxprob` as an escalation gate (lower AURC) | Margin is robust to overall logit scale | |
| P7 | Cascade matches uniform-16 accuracy at <=10 average frames | Combines P5 and P6 | |
| P8 | Oracle routing matches uniform-16 at <6 average frames | Only the flip set needs escalation | |

A prediction that fails is a finding, and gets a paragraph in the report explaining why —
that paragraph is usually the most interesting one in the writeup.
