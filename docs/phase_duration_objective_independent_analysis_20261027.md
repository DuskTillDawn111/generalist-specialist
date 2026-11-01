# Independent mathematical analysis of phase-duration fitting

This analysis reads the objective in `statistical_profiles.py` only. It reads no held-out run outcomes, launches no GPU work, and changes no predictor code. All statements of consistency below concern the statistical estimator under stated assumptions, not hardware identification or a capacity guarantee.

## The current objective has a specific duration estimand

For one fixed target `(family, task, input, target_phase)`, let `b > 0` be the fixed arithmetic solo phase baseline, `T_i > 0` its observed wall duration, and `dt_ij` its wall exposure to complete task/phase context `j`. Exposures partition the phase: `sum_j dt_ij = T_i`. Define

```
a_ij = dt_ij / T_i                  (observed wall-time fraction)
X_ij = dt_ij / b = T_i a_ij / b
h_i(s) = sum_j a_ij s_j / b         (reciprocal effective duration)
m_i(s) = 1 / h_i(s)                 (duration for fixed fractions)
```

The current unpenalized criterion is `mean((X s - 1)^2)`. With one context, it is `mean((T_i s/b - 1)^2)`, giving

```
s_hat = b * sum(T_i) / sum(T_i^2)
d_hat = b / s_hat = sum(T_i^2) / sum(T_i).
```

For independent observations with finite second moment, `d_hat` converges to `E[T^2]/E[T] = E[T] + Var(T)/E[T]`, not `E[T]`. This is duration size weighting introduced by the residual definition; it is not evidence that service is intrinsically that slow. A solo baseline equal to the true arithmetic mean does not remove the bias: fitting those same single-context observations again still yields this ratio. Correlated requests alter variance and effective sample size but not this algebraic finite-sample result.

The finite bound and ridge can change the result further. The current `mean(data loss) + alpha * mean((s-prior)^2)` has a fixed alpha; that prior influence does not vanish as sample count grows. Thus claims of asymptotic arithmetic-mean consistency require inactive bounds and vanishing penalty on identifiable parameters. If the mean duration lies outside the permitted duration interval, even the corrected objective is constrained to a boundary.

## Recommended first candidate: duration-corrected convex weighted least squares

Use the same complete-context matrix, positive speed bounds, hierarchy and UNKNOWN domain, but replace the data criterion by

```
L_IG(s) = mean_i [ (b/T_i) * (X_i s - 1)^2 ]
        = b * mean_i [ T_i * (h_i(s) - 1/T_i)^2 ].
```

The factor `b/T_i` makes the loss dimensionless. For a group with one baseline b, replacing it by `1/T_i` changes only an overall constant; if regularization is present, its strength must be defined relative to that constant explicitly. Implementation is the existing bounded linear solver with row scaling `sqrt(b/T_i)`: augment the ridge rows only after this weighted scaling. Do not confuse exposure weights with the number of independent runs.

With one context and no active ridge/bound,

```
L = mean[(T_i/b) s^2 - 2s + b/T_i]
dL/ds = 2 * mean(T_i)/b * s - 2
s_hat = b / mean(T_i),    d_hat = mean(T_i).
```

For one context with the original group-level ridge `alpha*(s-prior)^2`, the inactive-bound closed form becomes `s_hat=(1+alpha*prior)/(mean(T)/b+alpha)`. This is a useful separate optimizer check; it also makes the remaining prior bias explicit. Clipping this scalar solution to the original positive box gives the constrained scalar solution.

For a baseline fixed or consistently estimated from the training-only solo data, the duration estimate therefore converges to the arithmetic conditional mean in the single-context limit. Row weights are positive; the original convex quadratic and bound handling remain available. An explicit minimum-positive duration comes from valid timestamp resolution/collection qualification, not a value tuned on an independent holdout.

This is more than a heuristic reweighting. Up to terms not involving the model, inverse-Gaussian negative log likelihood with fixed shape parameter lambda and mean m is

```
(lambda/2) * (T-m)^2 / (T*m^2)
 = (lambda/2) * [T h^2 - 2h + 1/T],  h=1/m.
```

Our criterion is b times that deviance without its constant multiplier. The exact likelihood interpretation needs conditionally independent inverse-Gaussian durations with `Var(T | a) = m(a)^3/lambda`, fixed or externally modeled lambda, and covariates treated as exogenous. Those assumptions are not established by the measured exposures. The mean-targeting property itself is broader: for fixed a, the conditional expected criterion is minimized at `h(a)=1/E[T|a]` without assuming an inverse-Gaussian distribution. This remains a proper mean-targeting score when the reciprocal linear link can represent that conditional mean. Under misspecification it defines a weighted statistical projection, not the true conditional mean in every context.

## Second convex candidate: reciprocal-mean Gamma objective

Fit the same positive h with

```
L_Gamma(s) = mean_i [ T_i h_i(s) - log(b h_i(s)) ].
```

The b inside the logarithm only supplies a dimensionless argument; it is constant within a target group. This is fixed-shape Gamma negative log likelihood up to constants/scale, with `Var(T | a)=m(a)^2/k`; exponential duration is the special case k=1. The loss is convex in s because h is positive linear, with gradient `mean[a_i/b * (T_i - 1/h_i)]` and positive semidefinite Hessian `mean[a_i a_i'/(b^2 h_i^2)]`. Existing box bounds keep h strictly positive. SciPy bounded smooth optimization or a convex solver is required instead of the current direct quadratic least squares.

For a single context, its optimum is again `s_hat=b/mean(T)` exactly. Conditional expected loss is minimized at `m(a)=E[T|a]`, independently of a literal Gamma distribution, if that mean is represented by the reciprocal link. Compared with IG-WLS, its score does not place the additional `h_i` factor in the estimating equation. Which variance pattern better predicts a new run is a training cross-validation question; neither distribution should be declared physically true without supporting evidence.

For fixed observed fractions, both losses target the mean with different misspecification/variance weightings. Inverse Gaussian is the least implementation change and preserves a convex quadratic. Gamma is a credible second model family if the first candidate fails the predeclared training-CV objective; it is not an after-the-fact replacement chosen for a favorable independent result.

## Multi-context meaning and the identifiability boundary

The event simulator uses completion when integrated speed reaches one. For a hypothetical path whose wall fractions a are held fixed, that same equation gives the harmonic effective duration `m=b/(sum a_j s_j)`. Therefore the reciprocal model respects the work-integration interface. A naive linear duration regression `T ≈ sum a_j d_j`, with `d_j=b/s_j`, uses an arithmetic mixture instead and is incompatible with this integration model except in special cases. Weighting by wall fractions does not turn them into work fractions.

However, actual `a_ij` is reconstructed from the observed future phase duration and observed co-resident boundaries. It is not an input known before execution. Target speed and every other request's speed affect the cuts and a; random duration also changes exposure. Consequently fitting `T | observed a` is a retrospective conditional model. Calling its Gamma or IG criterion an *exact joint likelihood of the original executions* is unjustified. It is an exact conditional likelihood only under an additional exogenous-fractions sampling model; otherwise it is a conditional pseudo-likelihood/mean-targeting surrogate. It can be useful, but it does not by itself prove that event simulation with newly generated fractions has the correct unconditional mean.

The design only identifies combinations in the span of the exposure vectors. If contexts always occur together in almost fixed proportions, many speed vectors produce the same fitted phase durations. Ridge selects one, but does not identify its individual physical speeds. Report the weighted design rank, singular values/conditioning, context counts at the run and phase level, active bounds and sensitivity to priors. A single-context row can directly anchor its context; varied multi-context paths can add information; neither supports unseen task/phase combinations.

Keep each full co-resident family/task/phase context in the new model, rather than collapsing task identity or using a common average slowdown. Only observed target-handler full-context support may enable the existing explicitly labeled fallback, if retained. An unobserved target/input/phase or absent handler support remains UNKNOWN. Positivity/finite checks, `sum(dt)=T` validation and strict complete context identity are mandatory. Numerical `[.05,8]` speed bounds remain empirical regularization, with their original semantics; changing them must be a versioned training-only choice. They are not hardware bounds.

Separate tasks/phases with different baseline b may have different natural noise scales. Choose and freeze target weights, handler versus phase weights, and whether the aim is request-weighted or run-weighted accuracy. Normalizing `1/T` weights to have mean one does not alter an unpenalized group optimum but changes ridge strength; include that normalization in the model definition. Strong handler priors can reintroduce biased phase predictions even with a corrected phase loss, so correct/validate both hierarchy levels and select both penalties using training folds.

## Direct duration or whole-trace simulation loss

An ordinary duration target `mean((T_i - m_i(s))^2)` for fixed observed a also has the arithmetic-mean population optimum. It is nonlinear and generally nonconvex in positive speeds: the second derivative of `(T-1/h)^2` with respect to h can change sign. Bounded nonlinear least squares is implementable, but it loses the current global convex solution and still conditions on future observed fractions. Feasible GLS can reduce variance if a training-estimated conditional variance is correct; it does not cure exposure endogeneity or lack of rank. Any feasible-GLS variance estimate must be recomputed within each training fold, with floors fixed from training numerical requirements. Weighting absolute duration residuals by `1/T^2` would generally abandon arithmetic-mean targeting again.

Whole-trace event-simulation fitting instead reruns the declared training arrival/input trace with predicted phase positions. A candidate objective can combine absolute phase/handler duration error with complete-trace E2E/queue error, for example training-normalized squared errors with target/run weights fixed in advance. This aligns train and inference phase evolution and removes use of actual future context fractions at scoring time. It is the most direct way to test whether corrected speeds explain a new run's queue and tail behavior.

The costs are much larger: each parameter update requires full multi-request event rollouts, event order changes create nonsmooth/nonconvex loss, incomplete drain introduces censoring, and many contexts remain weakly identified. Finite-difference gradients scale poorly with context count; coordinate search, small parameter corrections, or differentiable event approximations introduce their own optimization/approximation decisions. One full-trace squared E2E loss also does not automatically target expected P95 or expected utility, which are nonlinear across requests. Deterministic point simulation cannot represent run-level variability merely by changing the duration loss.

A practical order is corrected convex IG-WLS first, Gamma reciprocal mean as a predeclared alternative, and simulator-based training-CV selection of these candidates. Reserve direct simulation parameter optimization for a separately versioned method if mean-corrected coefficients remain inadequate on training folds. Do not use per-phase quantiles as a substitute for cohort P95.

## Training-only model selection and honest evaluation

Use the existing whole-seed training split: leave one entire continuous-training seed out across all static candidates, refit every solo baseline, hierarchy coefficient, input/context support and any variance model from the remaining training sources, and freshly simulate every held-out training flow. Shared seed/input runs must remain grouped. Older profile sources may remain in every fold only if that rule is predeclared and none derive from a held-out continuous run. Never use actual held-out context boundaries as simulator inputs; they can be used afterward for descriptive phase error.

Predeclare a small candidate set: old objective as reference; IG-WLS and Gamma mean-link; a finite regularization grid including zero where supported; a deterministic rule for ties, unsupported contexts and failed/incomplete rollouts. Keep support domain, context encoding and positive speed bounds identical when comparing objective families. A concrete primary selection score is a request- and task-balanced training-fold squared worker-duration error normalized by training-only mean duration, averaged first within task and then across seeds. Add full-cohort E2E/P95 and ΔU ranking errors as separately reported training-CV endpoints, or predeclare their weights if selecting on them. Do not silently introduce a threshold after viewing a held-out outcome.

All candidates must report coverage. Either compare on a prospectively fixed common supported domain while separately reporting all missing predictions, or use a fixed failure policy that prevents an estimator winning by returning UNKNOWN on hard cases. Relative absolute error alone is not a mean-targeting selection rule: its optimum is a duration median weighted by `1/T`, generally different from the arithmetic mean. Report MARE for comparison if useful, while retaining a declared mean-targeting primary criterion. With only two seeds, cross-validation yields two correlated route blocks per objective and model choice is noisy; request-level resampling cannot manufacture independent seed repeats. If candidate penalties repeatedly hit a bound or outcomes are sensitive, report that uncertainty and retain the independent test as the actual confirmation.

Select the objective/hyperparameters using training folds only, refit once on all declared training sources, freeze source hashes and the complete event/metric policy, then evaluate on a new untouched run. Mean consistency in a single context is a unit-level mathematical property; reliable multi-context queue, P95, ΔU signs and candidate ranking remain independent empirical requirements.

## Independent numerical implementation review

The new separate `duration_profiles.py` and `background_duration.py` were read after their initial creation; the old engine was not edited. The weighted design uses `sqrt((1/sum(X))/n)` and retains `sqrt(alpha/p)` ridge rows, matching the declared dimensionless `w=b/T` criterion without weight renormalization. The background wrapper retains the bijective `(family,task)` encoding, full phase contexts, training-only role checks and old prediction interface.

A separate CPU calculation, without changing source files, checked:

| Check | Result |
| --- | --- |
| T=[100,200,500] ms, b=100 ms, alpha=0 | Predicted duration 266.6666666667 ms, exactly arithmetic sample mean |
| Same observations, alpha=.1, prior=1 | Speed .3975903614457833, equal to scalar ridge closed form |
| A non-diagonal rank-two, four-row exposure design | Solver speeds [.5397497966097176,.41402659411301784], equal to independently solved weighted normal equations |
| Very large/small normalized durations | Speed constrained to the declared [.05,8] interval |

These checks establish the numerical target and positivity behavior. They do not establish exogenous context fractions, multi-context physical identification, P95 accuracy, or independence of requests. The parent has separately chosen a predeclared lexicographic training decision rule based on ΔU sign/rank and original error gates; that is a decision-performance selection rule, not a theorem that the selected ridge predictor equals every conditional duration mean. Duration MSE and mean bias should still be reported descriptively.
