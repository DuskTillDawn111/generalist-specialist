"""LP calls use the same single-thread HiGHS scheduler as the Joint master."""

import warnings
from scipy.optimize import linprog as _linprog


def linprog(*args, **kwargs):
    # HiGHS has a process-wide scheduler. An LP initialized with its default
    # thread count can make a later original threads=1 MILP fail with status4.
    # Keep one convention without altering the historical MILP optimizer.
    kwargs['options']={**kwargs.get('options',{}),'threads':1}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected:.*threads.*')
        return _linprog(*args,**kwargs)
