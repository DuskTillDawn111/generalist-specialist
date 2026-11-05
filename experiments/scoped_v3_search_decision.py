"""Select among bounded v3 proposals without authorizing new measurements."""
import copy
import math


def summarize_selection(rows, review):
    if not rows or len({(r['method'], r['seed']) for r in rows}) != len(rows):
        raise ValueError('Unique nonempty search results required')
    if any(type(r['utility']) not in (int, float) or not math.isfinite(r['utility'])
           or type(r['matches_archived_pure_specialist']) is not bool for r in rows):
        raise ValueError('Finite feasible search results with explicit evidence coverage required')
    best = max(r['utility'] for r in rows)
    tied = [r for r in rows if r['utility'] == best]
    # Prefer evidence coverage only among exact objective ties, never replace
    # a better unmeasured proposal and claim it was the predicted winner.
    selected = min(tied, key=lambda r: (not r['matches_archived_pure_specialist'], r['method'], r['seed']))
    covered = selected['matches_archived_pure_specialist']
    return dict(selected_search_result=copy.deepcopy(selected), predicted_best_utility=best,
        all_method_winners_have_existing_measurements=all(r['matches_archived_pure_specialist'] for r in rows),
        selected_plan_has_existing_measurements=covered,
        recommendation='use_existing_pure_specialist_control' if covered else 'retain_measured_control_pending_unconfirmed_proposal',
        unconfirmed_predicted_winner=None if covered else copy.deepcopy(selected),
        selected_plan_archived_mean_U=(review['default']['assessment']['mean_U']['observed']['candidate-1'] if covered else None),
        archived_panel_mean_U=copy.deepcopy(review['default']['assessment']['mean_U']['observed']),
        independent_v3_confirmation=False, new_gpu_episodes=0,
        next_action='close_bounded_search_no_automatic_experiments')
