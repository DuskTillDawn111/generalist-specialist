"""Validated lifecycle query boundary for the frozen v24 full-risk fit."""
from gsdeploy.forecast_errors import UnknownEvidence
from .owner_lifecycle_mark_v21 import RESIDENTS, state_features
from .owner_lifecycle_hazard_v24_fit import LifecycleSaveHazard as FittedLifecycleSaveHazard


class LifecycleSaveHazard(FittedLifecycleSaveHazard):
    def query(self,held_owner,pending_frontier,owner_lifecycle,context_age_ns,horizon_ns=0):
        state_features(owner_lifecycle)
        ready={row[0] for row in pending_frontier}
        if not ready <= set(RESIDENTS):
            raise UnknownEvidence('FOREIGN_GPU_FRONTIER_OWNER')
        if any(owner_lifecycle[owner].gpu_ready is not (owner in ready) for owner in RESIDENTS):
            raise UnknownEvidence('OWNER_LIFECYCLE_READY_FRONTIER_MISMATCH')
        return super().query(held_owner,pending_frontier,owner_lifecycle,context_age_ns,horizon_ns)
