"""Read the complete static local-import closure from explicit entry files."""
import ast
from pathlib import Path

from .native_collect import receipt


def strict_references(value):
    """Deduplicate exact receipts, rejecting conflicting training identities."""
    found = {}

    def visit(item):
        if isinstance(item, dict):
            if set(item) == {'path', 'sha256'}:
                previous = found.get(item['path'])
                if previous is not None and previous != item:
                    raise ValueError('Conflicting implementation or training source receipts: '+item['path'])
                found[item['path']] = dict(item)
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return [found[path] for path in sorted(found)]


def implementation_sources(*entries):
    root = Path(__file__).resolve().parents[2]
    pending = []
    found = set()

    def enqueue(path):
        if path not in found:
            pending.append(path)
        # Importing a child module also executes the parent packages. Include
        # their initializers and recursively inspect their imports as well.
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent/'__init__.py'
            if initializer.is_file() and initializer not in found:
                pending.append(initializer)

    def add(parts):
        target = root.joinpath(*parts)
        for candidate in (target.with_suffix('.py'), target/'__init__.py'):
            if candidate.is_file() and candidate not in found:
                enqueue(candidate)

    for path in (*entries, __file__):
        enqueue(Path(path).resolve())

    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split('.')
                    if parts[0] in ('gsdeploy', 'tools'):
                        add(parts)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                parts = base + (node.module.split('.') if node.module else [])
                if parts and parts[0] in ('gsdeploy', 'tools'):
                    add(parts)
                    for alias in node.names:
                        if alias.name != '*':
                            add(parts+[alias.name])
    return [receipt(path) for path in sorted(found)]
