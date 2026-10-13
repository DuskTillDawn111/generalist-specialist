from .common import *
def main():
 sp=read(OUT/'candidate_space_definition.json');ds=read(OUT/'optimizer_decisions.json');names={'qwen':'Qwen','minicpm':'MiniCPM','phi4mm':'Phi','roberta_sst2':'Roberta','convnext_tiny':'ConvNext','whisper-small':'W.Small'}
 for m in sp['members']['C']:
  if 'whisper-medium@' in m:names[m]='W.Medium'
  elif 'whisper-large-v3@' in m:names[m]='W.Large'
  elif m.startswith('baudm/parseq:parseq@'):names[m]='PARSeq'
  elif m.startswith('textattack/'):names[m]='Bart'
 rows=[]
 for d in ds:
  r={k:d[k] for k in ['pool','scenario','algorithm','status']};r.update(structure=d.get('structure'),predicted_U=d.get('predicted_U'),regret=d.get('regret'))
  if 'deployment' in d:
   r['resident_replicas']='; '.join(f'{names[m]}×{n}' for m,n in d['deployment'].items());r.update({t:'; '.join(f'{names[m]}:{q:.8g}qps' for m,q in rr.items()) for t,rr in d['routing'].items()})
  rows.append(r)
 csvsave('deployment_selection_routes.csv',rows);save('model_display_aliases.json',names)
 print('ROUTE_TABLE',len(rows))
if __name__=='__main__':main()
