import json,sys,time
print('READY|'+json.dumps({'tasks':['text']}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r.get('op')=='shutdown':
  print('BYE|{}',flush=True);break
 start=time.time_ns();time.sleep(.001);end=time.time_ns()
 print('RESULT|'+json.dumps(dict(request_id=r['request_id'],task_type=r['task_type'],worker_latency_ms=(end-start)/1e6,queue_wait_ms=(start-r['submit_ns'])/1e6,start_ns=start,end_ns=end)),flush=True)
