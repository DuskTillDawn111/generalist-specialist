# twoFIFO paired transport partial identification v2

A busy transport component need not always remain UNKNOWN. The pure interface in
`twoFIFO_transport_partial_identification_v2.py` computes exactly what source
clocks identify, while leaving the unobserved stdin-line split unselected.

For actual submit `s`, parent write record `w`, worker START `b`, and previous
actual handler READY `r`, let `W=w-s`, `H=b-s`, and `q=r-s`. The complete line is
available at `s+u`, with `0 <= u <= min(W,H)`. Its paired post-release continuation
is exactly `x=H-max(q,u)`. This is a joint feasible set. Sampling its two margins
independently would violate the observed source START recurrence.

`identify_source(s,w,b,r)` returns this set and the exact continuation bounds.
If `r>=w`, every feasible line is ready before the previous READY and continuation
is point identified as `b-r`. The line itself remains unidentified. If `r<=s`,
only the composite `H=u+x` is identified in general. Missing previous actual READY
is retained as UNKNOWN; startup or an older handler is never silently substituted.

`project_target(source,s_target,r_target)` explicitly assumes that a complete
paired source transport vector transfers, including `W`, `u` and `x`. It computes
all feasible target starts:

```
START_target(u) = s_target + max(r_target-s_target,u) + H-max(q,u)
```

The exact interval comes from its two endpoints and the two piecewise linear
corners within the feasible interval. If source and target both have READY after
the respective write record, START is `r_target + (b-r)`. If source and target
both have READY before submit, START is `s_target + H`, even though line and
continuation are separately unknown. Other schedules can still be point
identified in special boundary cases; otherwise the interface returns the full
START envelope. It does not select zero line delay, fit a latent distribution,
claim an envelope probability, or qualify changed-schedule transfer.

The forensic runner preserves whole-attempt priming and every offered measured
row. For the first completed v2 shard, all 46 measured source STARTs reconstruct
exactly for every feasible paired decomposition: 13 have READY after write and
an identified continuation, 33 have READY before submit and an identified
composite handoff; none have READY inside submit/write. Of 18 priming rows,
four lack an earlier actual handler READY and remain UNKNOWN, while fourteen
have READY before submit. These counts describe the v2 contamination-HOLD
capture only. They are not production training support and are not donor laws.
The report records `source_training_domain_qualified=False` and retains the
explicit v2 HOLD status. A later qualified v3 compiler can use the pure feasible
set interface without admitting any old v2 observation.

Five CPU tests cover busy-to-busy and idle-to-idle identification, changed-regime
nontrivial envelopes, equality and late-write boundaries, rational clocks,
independent enumeration of all feasible grid points and corners, exact same
schedule reconstruction, invalid source chronology and unknown offered history.
