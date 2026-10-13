# Module 9BC

Development-only, bounded J-routing and candidate-ranking validation. Frozen v2
sources are reused through a private import namespace; only ranking is replaced.
Canonical feedback still defines the original neighborhood and capacity failures.
The optional routing runtime adapter substitutes J routes for actual serving
evaluation, recording the route separately. Worker/runtime sources are unchanged.

Use `/root/miniconda3/envs/gsdeploy/bin/python` with one BLAS/OpenMP thread.
`replay` freezes existing development inputs, audits historical states, validates
J solutions, replays candidates, and commits mechanical GPU selections. `validation`
checks analytic optima and identical v2 generation. `diagnostics` records evidence
limits and CPU overhead. The GPU launcher must inherit exactly the existing
`execution_environment_overrides` in the frozen method comparison protocol.
`launch` inherits that frozen environment and invokes `gpu` for the bounded plan;
completed validation cannot be relaunched through this entry point. `report`
verifies original inputs, freezes the
minimal default from the gates, writes all required answers, then stops.

The J objective has no fitted weights, no interference model, and no exact queueing
model claim. Runtime solver failure is distinct from capacity infeasibility.
The canonical capacity gate remains in candidate generation; J solve capacity is
rho < 1 with a numerical epsilon only. Original accept/rollback remains frozen.

After closeout, `module9bc.frozen_v3.create` reads the frozen gate selection and
constructs the chosen routing/ranking combination. `frozen_v3.snapshot` records
actual serving routes separately from canonical generation feedback and adds J
scores to the normal v2 trace. It does not change the neighborhood or evaluator.
