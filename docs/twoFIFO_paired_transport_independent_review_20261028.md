# Independent paired transport review

Let source submit be s, parent write-record time w, worker START b and previous
actual handler READY r. Define W=w-s, H=b-s and q=r-s. The transport contract
requires an unknown complete-line delay u and nonnegative continuation x with
0<=u<=W and H=max(q,u)+x. Consequently the exact feasible set is
0<=u<=min(W,H), x=H-max(q,u), provided H>=max(q,0).

If this same paired vector and W transfer to a target submit s' and generated
previous actual READY r', then target START is
s'+max(r'-s',u)+H-max(q,u). This is continuous and piecewise linear in u. Its
minimum and maximum occur at interval endpoints or either in-interval corner
q and r'-s'. The image is exactly the entire closed interval between those
extrema. Independent component bounds must not be sampled separately.

A singleton interval identifies target START conditionally without identifying
u or x. A nonsingleton interval cannot justify a point prediction. The isolated
v5.2 engine therefore calls the projection only after the previous handler's
READY has been generated, raises UNKNOWN with exact bounds and the full offered
cohort when needed, and never substitutes a line/continuation point. Sender
FIFO uses only the observed enqueue delay, sender turn and write duration.

Independent tests construct 300 feasible latent truths and check containment,
clock translation invariance, exact source-schedule reconstruction, attainable
paired extrema, closed canonical schemas, exact rational types, training-role
gates, later queued UNKNOWN after JSON continuation, initial message offsets,
and full denominator retention. Reviewed fixes add canonical source checking
and training-role checking before label inspection. Original collector source
closures and frozen v5/v5.1 remain unchanged.

This is a conditional identification result under an explicit paired transport
transfer assumption. It does not identify a latent probability law or certify
changed-schedule native transfer, the response model, prediction accuracy,
resource predictions, or final routing benefit. Historical perturbed v2 capture
remains HOLD and supplies no qualified training support.
