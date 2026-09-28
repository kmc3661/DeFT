"""Read-only original-token identity tracing; disabled in normal inference."""
from types import MethodType


def capture_trace(model):
    state=model._progressive
    trace=[]
    begin=state.begin
    commit=state.commit_physical_compaction
    def traced_begin(self,*a,**kw):
        value=begin(*a,**kw)
        trace.append({})
        return value
    def traced_commit(self,keep_idx):
        if self._physical_pending_kind=='final':
            visual=self._vision_by_batch[0]
            mask=self._vision_keep[0,visual]
            candidates=self._feature_reserve_original_local_indices
            trace[-1]['candidates']=candidates.detach().cpu().tolist()
            trace[-1]['K_original']=candidates[mask].detach().cpu().tolist()
        return commit(keep_idx)
    state.begin=MethodType(traced_begin,state)
    state.commit_physical_compaction=MethodType(traced_commit,state)
    return trace
