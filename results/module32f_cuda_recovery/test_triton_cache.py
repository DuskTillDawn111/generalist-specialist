"""CPU reproduction of ABI1 EXDEV and byte/atomicity cache regression."""
import sys,threading,hashlib,json
from m32f.isolation import install,VIOLATIONS,ROOT
layer=sys.argv[1];install(layer)
from triton.runtime.cache import FileCacheManager
from m32f.triton_cache import SameDirectoryCacheManager
original=FileCacheManager('CPU_ORIGINAL_'+layer);errors=[];result=dict(layer=layer)
try:original.put(b'identical native cache bytes','kernel.bin');result['original_put_success']=True
except OSError as exc:result.update(original_put_success=False,original_errno=exc.errno,original_error=repr(exc))
manager=SameDirectoryCacheManager('CPU_REPAIRED_'+layer)
values=[b'identical native cache bytes','{"metadata": 123}']
for i,data in enumerate(values):
 path=manager.put(data,f'kernel_{i}.bin',binary=True)
 expected=data if isinstance(data,bytes) else str(data).encode()
 assert open(path,'rb').read()==expected
group={'bin':manager.get_file('kernel_0.bin'),'meta':manager.get_file('kernel_1.bin')}
manager.put_group('kernel.json',group);assert manager.get_group('kernel.json')==group
payloads=[bytes([i])*65536 for i in range(12)];manager.put(payloads[0],'contended.bin');start=threading.Event()
def writer(i):
 try:
  start.wait()
  for _ in range(8):manager.put(payloads[i],'contended.bin')
 except BaseException as exc:errors.append(repr(exc))
threads=[threading.Thread(target=writer,args=(i,)) for i in range(12)]
for t in threads:t.start()
start.set();observed=0
while any(t.is_alive() for t in threads):
 data=open(manager.get_file('contended.bin'),'rb').read();assert data in payloads,'Partial cache publication';observed+=1
for t in threads:t.join()
result.update(passed=not errors and not VIOLATIONS,cache_bytes_sha256=[hashlib.sha256(v if isinstance(v,bytes) else v.encode()).hexdigest() for v in values],group_exact=True,concurrent_writers=12,atomic_reader_observations=observed,errors=errors,unexpected_writes=VIOLATIONS)
(ROOT/f'triton_cache_CPU_{layer}.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['passed']
if layer=='repaired':assert not result['original_put_success'] and result['original_errno']==18
else:assert result['original_put_success']
print('CACHE_CPU_PASS',layer,result,flush=True)
