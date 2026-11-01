"""Inspect relevant module processes without resolving unrelated process cwd."""
from pathlib import Path


def identity(pid, *, proc_root=Path('/proc')):
    folder=proc_root/str(pid)
    try:
        stat=(folder/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        argv=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=pid,start_ticks=int(fields[19]),argv=argv,cwd=str((folder/'cwd').resolve()))
    except (FileNotFoundError,ProcessLookupError):return None


def modules(module, *, proc_root=Path('/proc')):
    result=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():continue
        try:argv=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError,ProcessLookupError):continue
        # Cwd can be unreadable during unrelated executable transitions. It is
        # required only for an actual candidate running this exact module.
        if not any(a=='-m' and i+1<len(argv) and argv[i+1]==module for i,a in enumerate(argv)):continue
        value=identity(int(folder.name),proc_root=proc_root)
        if value is not None:result.append(value)
    return result
