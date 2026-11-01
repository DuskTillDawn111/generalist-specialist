# Current-frontier next-owner candidate v14

v14 removes every full-future-workload feature from v13. Its 90 features use
current held owner, continuous attained RESTORE age, current pending
owner/task/pool/semantic phases, and the previous observed or modelled held owner
with an explicit missing flag. Optional `public_workload` input validates public
identities and declared offsets, then discards them. Future requests, counts,
composition and offsets are advanced by the DAG and cannot alter this mark.
The current frontier follows the runtime FIFO contract: at most one running
request per owner. Public pool identities must be among the training vocabulary.

The training cohort remains all 35,737 transitions in two original captures,
including all 715 targets outside the pending frontier. Previous-holder history
is linked only when one prior transition in the same capture has exactly the
current RESTORE timestamp and current owner as its next RESTORE and next owner.
Its SAVE must already have happened. Of the training rows, 35,673 have a unique
link and 64 remain missing. Ambiguous matches also remain missing. No history is
joined across captures or synthesized from a future target.

A new simulated initial state does not inherit the original capture's prior
history. For the 27 paired source initial-state candidates, `previous_held_owner`
defaults to `None` unless the runtime independently has observed/modelled history.
A missing-history warning remains explicit. A mark prediction is conditional on
SAVE now and does not identify SAVE, restore, service-ready time, or SM service.

```python
from tools.predictor_validation_20261027.context_next_owner_v14 import NextOwnerCandidate
candidate = NextOwnerCandidate(model)
result = candidate.predict_given_save_now(
    held_owner=held_owner,
    context_age_ns=attained_age_ns,
    pending_frontier=current_frontier,
    previous_held_owner=previous_observed_or_modelled_owner,  # None when absent
)
```

The fixed L2 multinomial objective retains the v13 lambda=1 convention:
`regularization / training_transition_count` times the squared coefficient
penalty, including the intercept. Captures receive equal loss weight. Complete
raw-SHA captures are held out; feature scaling, ranges, fitted coefficients and
all majority/lookup baselines use only the training fold. No hyperparameters are
selected from these folds. The no-history ablation removes all five history
features and is fitted separately. A second diagnostic uses the history model
but supplies `None` for every test previous owner.

The completed artifact is
`artifacts/predictor-context-next-owner-v14-20261028-training02/`.
All five fitted optimizations converged; full fit needed 338 iterations. The
first run reached the old 300-iteration solver budget. That diagnostic and a
copy of its implementation remain in `training01`; only the optimization budget
was increased to 1,000 for training02. The objective, lambda and feature design
were unchanged.

| Whole-capture training diagnostic | Capture 26305a… | Capture 98c4c2… |
| --- | ---: | ---: |
| Candidate with source-linked history | 99.8301% | 99.8783% |
| Independently fitted no-history ablation | 99.3772% | 99.3748% |
| History model with all previous owners missing | 98.7827% | 99.0373% |
| Training holder-only majority | 59.6592% | 59.5408% |
| Holder/frontier lookup with holder/global fallback | 99.3093% | 99.3638% |
| Holder/frontier/previous lookup with holder/global fallback | 99.7792% | 99.8174% |

Every baseline covers the full test cohort. The naturally missing-history
subgroups have only 33 and 31 observations, with 32 and 29 correct. OCR has only
two target observations; this cannot support reliable OCR-specific accuracy.
These are dependent training diagnostics from two repeated workload captures,
not independent accuracy, calibration, throughput or deployment-utility results.

Support warnings cover marginal feature ranges, holder-specific age ranges, and
unseen joint holder/current-frontier/previous-owner combinations. Changing a
valid future cohort to 4×57 requests, changing all offsets, or reversing its
order leaves actual-model features, probabilities and OOD flags identical.
That does not establish support for a changed current frontier. For example,
current OCR pending work and generalist image work are absent from these
transition features and correctly trigger support warnings.

`tests/test_context_next_owner_v14.py` has nine CPU tests for same-capture unique
history, missing/ambiguous history, future-cohort invariance, source binding,
causal input rejection, explicit OOD, baseline fallback, and complete-capture
training/artifact generation. An actual-data verification also checked every
history link, all prediction/cohort accounting, and the 4×57 invariance.
