"""Module 3A: non-enumerative, LP-guided binary deployment search."""
from .feedback import BottleneckFeedback
from .result import SearchResult
from .routing import CanonicalRouting, solve_canonical_routing
from .runtime import RuntimeEvaluation, RuntimeEvaluator, SyntheticRuntimeEvaluator, weighted_task_latency
from .scoring import PerformanceCoarseScorer, RuntimeObjective
from .search import GuidedDeploymentSearch, SearchConfig
from .integer_search import IntegerGuidedDeploymentSearch, IntegerSearchResult
from .routing_search import GuidedRoutingSearch, RoutingSearchConfig, RoutingSearchResult
from .dynamic_adaptation import (DynamicAdaptation, AdaptationConfig, AdaptationHistory,
    DynamicSearchResult, DynamicState, ReconfigurationCost, ReconfigurationCostModel, StructuralReconfigurationCost)

__all__ = ['BottleneckFeedback', 'SearchResult', 'CanonicalRouting', 'solve_canonical_routing',
           'RuntimeEvaluation', 'RuntimeEvaluator', 'SyntheticRuntimeEvaluator', 'weighted_task_latency',
           'PerformanceCoarseScorer', 'RuntimeObjective', 'GuidedDeploymentSearch', 'SearchConfig',
           'GuidedRoutingSearch', 'RoutingSearchConfig', 'RoutingSearchResult',
           'DynamicAdaptation', 'AdaptationConfig', 'AdaptationHistory', 'DynamicSearchResult', 'DynamicState',
           'ReconfigurationCost', 'ReconfigurationCostModel', 'StructuralReconfigurationCost',
           'IntegerGuidedDeploymentSearch', 'IntegerSearchResult']
