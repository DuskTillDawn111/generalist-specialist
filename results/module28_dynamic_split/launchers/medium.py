import ast,sys
from pathlib import Path
sys.argv=['/root/autodl-tmp/generalist-specialist/module13b/worker.py', '--model-id', 'openai/whisper-medium', '--task', 'asr', '--directory', '/root/autodl-tmp/generalist-specialist/models/module13b/openai--whisper-medium']
p=Path(sys.argv[0]);tree=ast.parse(p.read_text())
class Dispatch(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  if node.name!='handler':return node
  for child in ast.walk(node):
   if isinstance(child,ast.Attribute) and isinstance(child.value,ast.Name) and child.value.id=='args' and child.attr=='task':
    child.value.id='request_task_namespace'
  node.body.insert(0,ast.parse("request_task_namespace=type('Task',(),{'task':request['task_type']})()").body[0])
  return node
for node in ast.walk(tree):
 if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='run_worker':
  node.value.args[2]=ast.parse("dict(metadata,tasks=['asr','lid'])",mode='eval').body
namespace={'__file__':str(p),'__name__':'__main__'}
tree=Dispatch().visit(tree);ast.fix_missing_locations(tree)
exec(compile(tree,str(p),'exec'),namespace)
