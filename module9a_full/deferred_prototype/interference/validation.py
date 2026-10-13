"""Pair-identity statistics; repetitions never cross evidence splits."""
import math
import numpy as np
from scipy.stats import spearmanr,kendalltau

def metrics(predicted,real):
 p=np.asarray(predicted,float);r=np.asarray(real,float);error=p-r
 if len(p)==0:return {'n_pairs':0,'MAE':None,'median_AE':None,'RMSE':None,'MAPE':None,'Spearman':None,'Kendall':None,'ranking_agreement':None}
 sensible=np.abs(r)>np.sqrt(np.finfo(float).eps)
 def finite(v):return float(v) if math.isfinite(v) else None
 varied=len(p)>1 and len(set(p))>1 and len(set(r))>1
 signs=[(np.sign(p[i]-p[j]),np.sign(r[i]-r[j])) for i in range(len(p)) for j in range(i)]
 comparable=[(a,b) for a,b in signs if a!=0 and b!=0]
 return {'n_pairs':len(p),'MAE':float(np.mean(np.abs(error))),'median_AE':float(np.median(np.abs(error))),'RMSE':float(np.sqrt(np.mean(error**2))),'MAPE':float(np.mean(np.abs(error[sensible]/r[sensible]))*100) if np.any(sensible) else None,'MAPE_numerical_exclusion_count':int(np.sum(~sensible)),'MAPE_caveat':'dimensionless slowdown increment mu can be close to zero; also inspect MAE','Spearman':finite(spearmanr(p,r).statistic) if varied else None,'Kendall':finite(kendalltau(p,r).statistic) if varied else None,'ranking_agreement':float(np.mean([a==b for a,b in comparable])) if comparable else None,'ranking_comparable_pairs':len(comparable),'ranking_tie_pairs':len(signs)-len(comparable)}

def paired_model_bootstrap(real,i1,i2,resamples=10000,seed=94001):
 r=np.asarray(real,float);a=np.asarray(i1,float);b=np.asarray(i2,float)
 if not len(r):raise ValueError('no held-out pairs')
 differences=np.abs(b-r)-np.abs(a-r)
 rng=np.random.default_rng(seed);boot=differences[rng.integers(0,len(r),size=(resamples,len(r)))].mean(axis=1);ci=np.quantile(boot,[.025,.975])
 verdict='TASK_PATH_PREDICTOR_IMPROVES_OVER_TASK_AGNOSTIC' if ci[1]<0 else 'NOT_SUPPORTED' if ci[0]>0 else 'INCONCLUSIVE'
 return {'metric':'pair-level mean absolute error(I2)-absolute error(I1)','n_validation_pair_identities':len(r),'point_difference':float(differences.mean()),'CI95':ci.tolist(),'bootstrap_seed':seed,'resamples':resamples,'bootstrap_unit':'directional pair identity; all repetitions summarized within identity','per_pair_error_difference':differences.tolist(),'decision':verdict,'TASK_PATH_PREDICTOR_IMPROVES_OVER_TASK_AGNOSTIC':True if ci[1]<0 else False if ci[0]>0 else None,'fixed_percent_improvement_threshold':None}

def slowdown_bootstrap(standalone,pair,resamples=10000,seed=94002):
 a=np.asarray(standalone,float);b=np.asarray(pair,float)
 if a.shape!=b.shape or a.ndim!=2 or a.shape[0]<3:raise ValueError('matched >=3 independent process blocks required')
 rng=np.random.default_rng(seed);n,q=a.shape;blocks=rng.integers(n,size=(resamples,n));samples=rng.integers(q,size=(resamples,n,q))
 ba=a[blocks[:,:,None],samples].mean(axis=2);bb=b[blocks[:,:,None],samples].mean(axis=2)
 # Aggregate protocol: ratio of pooled means for the identity; block hierarchy retained.
 boot=bb.mean(axis=1)/ba.mean(axis=1);ratio=float(b.mean()/a.mean());ci=np.quantile(boot,[.025,.975])
 return {'S_real':ratio,'mu_real':ratio-1,'S_real_CI95':ci.tolist(),'mu_real_CI95':(ci-1).tolist(),'run_S':(b.mean(axis=1)/a.mean(axis=1)).tolist(),'protocol':'AGGREGATE_PROTOCOL_RETAINED','estimand':'mean(pair latency)/mean(standalone latency) pooled over exact matched independent repetitions','resamples':resamples,'seed':seed}
