import ast,sys
from pathlib import Path
sys.argv=['/root/autodl-tmp/generalist-specialist/module14b/worker.py', '--model-id', 'baudm/parseq:trba']
p=Path(sys.argv[0]);tree=ast.parse(p.read_text())
for node in ast.walk(tree):
 if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='run_worker':
  node.value.args[2]=ast.parse("dict(metadata,tasks=['ocr'])",mode='eval').body
ast.fix_missing_locations(tree)
exec(compile(tree,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
