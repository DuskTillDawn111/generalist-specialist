"""Reuse the existing convex J certificate with numerical cut reuse only."""
import ast, inspect, textwrap, warnings, math
from scipy.optimize import milp
from module11b import convex_audit as original


def precise_milp(*args, **kwargs):
    kwargs['options'] = {**kwargs.get('options', {}), 'threads': 1, 'random_seed': 0,
        'mip_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9,
        'dual_feasibility_tolerance': 1e-9}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
        return milp(*args, **kwargs)


class CutReuse(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name != 'solve_discrete_J': return node
        out = []
        for statement in node.body:
            out.append(statement)
            if isinstance(statement, ast.For) and ast.unparse(statement).startswith('for m in models:\n    cut(m, 0)'):
                out += ast.parse('for m in models:\n    for rho in sorted(NUMERICAL_CUT_POOL.get(m, set())):\n        cut(m, rho)').body
        node.body = out
        # Persist only valid universal perspective tangents, not outcomes/routes.
        class Returns(ast.NodeTransformer):
            def visit_FunctionDef(self, item):
                return self.generic_visit(item) if item.name == 'solve_discrete_J' else item
            def visit_Return(self, item):
                return ast.parse('for name, values in cuts.items():\n    NUMERICAL_CUT_POOL.setdefault(name, set()).update(values)').body + [item]
        node = Returns().visit(node)
        return node


class StructuralSolver:
    def __init__(self):
        self.cuts = {}
        source = ast.parse(textwrap.dedent(inspect.getsource(original.solve_discrete_J)))
        source = CutReuse().visit(source); ast.fix_missing_locations(source)
        self.namespace = {**original.__dict__, 'milp': precise_milp, 'NUMERICAL_CUT_POOL': self.cuts}
        exec(compile(source, '<module12 numerical cut reuse>', 'exec'), self.namespace)
        self.function = self.namespace['solve_discrete_J']

    def solve(self, rates, catalog, roles, cal, budget):
        # Zero-rate tasks have exact zero routing; exclude their unnecessary
        # normalized optimization variables to avoid division by zero.
        positive = {k: v for k, v in rates.items() if v > 0}
        max_G = sum(cal.evidence_bounds[m] for m in catalog if roles[m] == 'generalist')
        result = self.function(positive, catalog, roles, cal, budget, max_G)
        result['evidence_scope'] = 'COUNTERFACTUAL_CPU_ONLY_MEASURED_COUNT_SPACE'
        if result['J_seconds'] is not None:
            # Exact-zero demand residents are objective-identical, unnecessary
            # MILP ties. Remove them without a penalty or a new objective.
            original_x = dict(result['deployment'])
            idle = [m for m, n in original_x.items() if n and
                    math.fsum(v for path, v in result['routing'].items() if path.split('::')[1] == m) == 0.]
            result['raw_unpruned_deployment'] = original_x
            result['exact_zero_demand_residents_removed'] = idle
            for m in idle:
                result['deployment'][m] = 0
                result['D'].pop(m, None); result['rho'].pop(m, None)
            result['routing'] = {path: v for path, v in result['routing'].items()
                                 if result['deployment'][path.split('::')[1]]}
            result['N_G'] = sum(n for m, n in result['deployment'].items() if roles[m] == 'generalist')
            result['safe_memory_mib'] = original.factor() * math.fsum(cal.memory_table_mib[m][n] for m, n in result['deployment'].items())
            result['max_rho'] = max(result['rho'].values(), default=0.)
            assert result['safe_memory_mib'] <= budget + 1e-6
            assert result['certificate']['passed'] and result['global_gap_seconds'] <= 1e-8
            for k, rate in rates.items():
                routed = math.fsum(v for path, v in result['routing'].items() if path.split('::')[0] == k)
                assert math.isclose(routed, rate, rel_tol=1e-7, abs_tol=1e-9)
        return result
