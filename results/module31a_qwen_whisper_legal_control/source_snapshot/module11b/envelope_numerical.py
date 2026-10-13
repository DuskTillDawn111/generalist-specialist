"""Same Module11A formulation with tighter HiGHS numerical feasibility checks.

The original source and its lexicographic objectives/constraints are untouched.
HiGHS' default MIP feasibility tolerance (1e-6) exceeds the original 1e-8
alpha tie interval. Tightening numerical tolerances avoids inconsistent later
tie-break stages; this never relaxes capacity, memory or quality constraints.
"""
import types
import warnings
from scipy.optimize import milp
from module11 import envelope


def _precise_milp(*args, **kwargs):
    kwargs['options'] = {**kwargs.get('options', {}),
        'mip_feasibility_tolerance': 1e-9,
        'primal_feasibility_tolerance': 1e-9,
        'dual_feasibility_tolerance': 1e-9}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
        return milp(*args, **kwargs)


_same_formulation = types.FunctionType(envelope.solve.__code__,
    {**envelope.solve.__globals__, 'milp': _precise_milp},
    name='same_formulation_with_precise_feasibility')


def solve(*args, **kwargs):
    result = _same_formulation(*args, **kwargs)
    result['numerical_solver_profile'] = 'HiGHS MIP/primal/dual feasibility=1e-9; same frozen formulation, objectives and tie intervals'
    return result


fixed_lp = envelope.fixed_lp
