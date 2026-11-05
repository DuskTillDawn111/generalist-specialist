"""External timing probes around unchanged optimizer sources.

Only diagnostic runs replace methods in memory; all replacements are restored.
AST probes mark phases, without changing any expression, constraint or branch.
Uninstrumented runs supply the primary timing comparisons.
"""

import ast
from contextlib import contextmanager
import functools
import inspect
import math
import textwrap
import time
from unittest.mock import patch


class PhaseClock:
    def __init__(self):
        self.stack = ['other_python']
        self.seconds = {}
        self.rounds = []
        self.round = None
        self.route = None
        self.started = self.last = time.perf_counter()

    def flush(self):
        now=time.perf_counter()
        delta=now-self.last
        self.last=now
        key=self.stack[-1]
        self.seconds[key]=self.seconds.get(key,0)+delta
        if self.round is not None:
            d=self.round['seconds']
            d[key]=d.get(key,0)+delta

    def mark(self, category):
        self.flush()
        self.stack[-1]=category

    @contextmanager
    def phase(self, category):
        self.flush()
        self.stack.append(category)
        try:
            yield
        finally:
            self.flush()
            self.stack.pop()

    def begin_round(self):
        self.flush()
        if self.round is not None:
            self.round['wall_seconds']=self.last-self.round.pop('_start')
        self.round=dict(route=self.route,iteration=1+sum(r['route']==self.route for r in self.rounds),
                        seconds={},prediction_bundles=0,identity_exclusions_added=0,
                        safely_pruned_deployments=0,_start=self.last)
        self.rounds.append(self.round)
        self.mark('milp_build')

    def end_round(self):
        self.flush()
        if self.round is not None:
            self.round['wall_seconds']=self.last-self.round.pop('_start')
            self.round=None

    def report(self):
        self.end_round()
        elapsed=self.last-self.started
        total=math.fsum(self.seconds.values())
        return dict(wall_seconds=elapsed,seconds=self.seconds,
            percent={k:100*v/elapsed for k,v in self.seconds.items()},
            accounted_seconds=total,accounting_error_seconds=elapsed-total,rounds=self.rounds)


def instrument_solve(original, clock, joint=False):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    func=tree.body[0]
    def probe(method,*args):
        return ast.Expr(ast.Call(ast.Attribute(ast.Name('_diagnostic_clock',ast.Load()),method,ast.Load()),
                                 [ast.Constant(a) for a in args],[]))
    class Probes(ast.NodeTransformer):
        def visit_Assign(self,node):
            # Phase switches after the original solver return, before feedback.
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='solve_master':
                return [node,probe('mark','feedback')]
            return node
        def visit_For(self,node):
            node=self.generic_visit(node)
            if joint and isinstance(node.target,ast.Name) and node.target.id=='iteration':
                node.body.insert(0,probe('begin_round'))
            return node
        def visit_While(self,node):
            node=self.generic_visit(node)
            if not joint:
                node.body.insert(0,probe('begin_round'))
            return node
    func=Probes().visit(func)
    func.body.insert(0,probe('mark','milp_build'))
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,_diagnostic_clock=clock)
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    return namespace[original.__name__]


@contextmanager
def diagnostics(clock):
    import scipy.optimize._milp as scipy_milp
    import gsdeploy.milp as joint
    import gsdeploy.optimizer as dfs
    import gsdeploy.continuous.predictor_adapter as adapter
    import gsdeploy.continuous.evaluator as evaluator
    import gsdeploy.continuous.de_optimizer as de
    import gsdeploy.continuous.fixed_routing_milp as fixed
    from contextlib import ExitStack

    def measured(function, category, prediction=False):
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            with clock.phase(category):
                if prediction and clock.round is not None:
                    clock.round['prediction_bundles']+=1
                return function(*args,**kwargs)
        return wrapped

    def master(function):
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            with clock.phase('milp_build'):
                row=clock.round
                con=kwargs.get('constraints')
                if row is not None:
                    row.update(rows=con.A.shape[0],columns=con.A.shape[1],nonzeros=con.A.nnz)
                start=time.perf_counter()
                fit=function(*args,**kwargs)
                if row is not None:
                    selected=tuple(i for i,v in enumerate(fit.x) if v>.5) if fit.x is not None else ()
                    row.update(solver_status=int(fit.status),solve_api_seconds=time.perf_counter()-start,
                               selected_variables=list(selected),dual_bound=float(fit.mip_dual_bound)
                               if getattr(fit,'mip_dual_bound',None) is not None and math.isfinite(fit.mip_dual_bound) else None)
                return fit
        return wrapped

    old_eval=evaluator.ContinuousRoutingEvaluator.evaluate
    def evaluate(self, *args,**kwargs):
        previous=clock.route
        clock.route=len(self.history)
        try:
            with clock.phase('other_python'):
                return old_eval(self,*args,**kwargs)
        finally:
            clock.route=previous

    old_fixed=fixed.FixedRoutingMilpOptimizer.solve
    old_joint=joint.JointMilpOptimizer.solve
    with ExitStack() as stack:
        def replace(obj,name,value):
            stack.enter_context(patch.object(obj,name,value))
        replace(scipy_milp,'_highs_wrapper',measured(scipy_milp._highs_wrapper,'highs_solver'))
        replace(joint,'solve_master',master(joint.solve_master))
        replace(fixed,'solve_master',master(fixed.solve_master))
        for module in (joint,dfs,adapter):
            replace(module,'predict_scenarios',measured(module.predict_scenarios,'predictor',True))
        replace(evaluator.ContinuousRoutingEvaluator,'check',measured(evaluator.ContinuousRoutingEvaluator.check,'identity_validation'))
        replace(evaluator.ContinuousRoutingEvaluator,'candidate',measured(evaluator.ContinuousRoutingEvaluator.candidate,'other_python'))
        replace(evaluator.ContinuousRoutingEvaluator,'evaluate',evaluate)
        replace(de.DERoutingOptimizer,'solve',measured(de.DERoutingOptimizer.solve,'outer_optimizer'))
        for cls,original,is_joint in [(fixed.FixedRoutingMilpOptimizer,old_fixed,False),(joint.JointMilpOptimizer,old_joint,True)]:
            compiled=instrument_solve(original,clock,is_joint)
            def make_solver(function):
                def wrapped(self,*args,**kwargs):
                    with clock.phase('milp_build'):
                        try:
                            output=function(self,*args,**kwargs)
                            # Fixed solver adds exactly one identity no-good after
                            # each scored witness. Final bound pruning is separate.
                            if 'ledger' in output:
                                for row in clock.rounds:
                                    if row['route']==clock.route and row['prediction_bundles']:
                                        row['identity_exclusions_added']=1
                                if clock.round is not None:
                                    clock.round['safely_pruned_deployments']=output['safely_pruned_deployments']
                                output['diagnostic_round_pruned_total']=output['safely_pruned_deployments']
                            return output
                        finally:
                            clock.end_round()
                return wrapped
            replace(cls,'solve',make_solver(compiled))
        yield
