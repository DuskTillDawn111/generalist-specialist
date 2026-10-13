import ast,sys
from pathlib import Path
sys.argv=['/root/autodl-tmp/generalist-specialist/module13b/worker.py', '--model-id', 'facebook/mms-lid-256', '--task', 'lid', '--directory', '/root/autodl-tmp/generalist-specialist/models/module13b/facebook--mms-lid-256']
p=Path(sys.argv[0]);tree=ast.parse(p.read_text())
for node in ast.walk(tree):
 if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='run_worker':
  node.value.args[2]=ast.parse("dict(metadata,tasks=['lid'])",mode='eval').body
ast.fix_missing_locations(tree)
exec(compile(tree,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
