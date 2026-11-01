"""Optional frozen risk at a reader cutoff, separate from six-term forecasts.

No labels or offline fitting enter this path. Runtime records are projected to
already received START/READY facts before feature construction. The caller's
original whole-horizon forecast and stability decision remain unchanged.
"""
from __future__ import annotations

import copy

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.recovery_risk import RecoveryRiskProvider
from gsdeploy.resources import identity
from .native_queue_risk_rows_v9 import public_trace,features,visible_projection


def predict_cutoff(plan,records,*,phase_start_ns,now_ns,model,cost_provider):
    context=plan['execution_context_sha256']
    if type(phase_start_ns) is not int or type(now_ns) is not int or not 0<=phase_start_ns<=now_ns:
        raise ValueError('Current cutoff in original native clock required')
    provider=RecoveryRiskProvider(model,context)
    try:
        current=features(public_trace(plan,phase_start_ns),visible_projection(records,now_ns),
            now_ns=now_ns,declared_at_ns=phase_start_ns,context=context,cost_provider=cost_provider)
        prediction=provider.predict(current,now_ns)
    except UnknownEvidence as exc:
        return dict(status='UNKNOWN',reason=str(exc),execution_context_sha256=context,
            cutoff_ns=now_ns,model_sha256=model['model_sha256'],six_term_utility_changed=False,
            stability_rule_changed=False)
    return dict(status='FROZEN_UNCALIBRATED_RISK_DIAGNOSTIC',execution_context_sha256=context,
        cutoff_ns=now_ns,feature_sha256=identity(current),features=current,prediction=prediction,
        labels_read=False,model_fitted_online=False,six_term_utility_changed=False,
        stability_rule_changed=False)


def attach_diagnostic(forecast,risk):
    """Attach an explanation even when the main DAG continuation is UNKNOWN."""
    if 'optional_recovery_risk' in forecast:
        raise ValueError('Do not overwrite a previously attached risk diagnostic')
    result=copy.deepcopy(forecast)
    result['optional_recovery_risk']=copy.deepcopy(risk)
    return result
