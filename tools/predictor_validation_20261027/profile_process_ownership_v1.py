"""Linux subreaper boundary for descendants that create new sessions/groups."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import signal
import time

NONCE_ENV = 'GS_PROFILE_TRAINING_EXECUTION_NONCE'


def subreaper(enable=None):
    libc=ctypes.CDLL(None,use_errno=True);flag=ctypes.c_int()
    if libc.prctl(37,ctypes.byref(flag),0,0,0):
        raise OSError(ctypes.get_errno(),'PR_GET_CHILD_SUBREAPER failed')
    previous=bool(flag.value)
    if enable is not None and libc.prctl(36,int(enable),0,0,0):
        raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
    return previous


def process_table():
    rows={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(path.name)]=dict(pid=int(path.name),state=fields[0],ppid=int(fields[1]),
                process_group=int(fields[2]),session=int(fields[3]),start_ticks=int(fields[19]))
        except (OSError,ValueError,IndexError):continue
    return rows


def ancestors_include(pid, ancestor, table):
    seen=set()
    while pid in table and pid not in seen:
        if pid==ancestor:return True
        seen.add(pid);pid=table[pid]['ppid']
    return False


class OwnedDescendants:
    """Single-purpose supervisor launches one collection, with prior trees excluded.

    Subreaping makes even a double-fork/setsid descendant observable after its
    parent exits. This dedicated supervisor must start without existing child
    trees: otherwise a later orphan from such a tree is ambiguous after its
    original parent exits. Refuse before collection rather than guess ownership.
    """
    def __init__(self,nonce):
        self.parent=os.getpid();self.nonce=nonce;self.root=None;self.root_key=None;self.known={};self.reaped=[]
        table=process_table()
        self.baseline={(pid,r['start_ticks']) for pid,r in table.items()
            if pid!=self.parent and ancestors_include(pid,self.parent,table)}
        if self.baseline:
            raise ValueError('Dedicated profile supervisor requires no pre-existing child trees')

    def observe(self,root=None):
        if root is not None:self.root=root
        table=process_table();found=[]
        if root is not None and root in table:self.root_key=(root,table[root]['start_ticks'])
        root_live=(self.root_key is not None and self.root in table
            and (self.root,table[self.root]['start_ticks'])==self.root_key)
        for pid,row in table.items():
            key=(pid,row['start_ticks'])
            if pid==self.parent or key in self.baseline:continue
            if any(base_pid in table and table[base_pid]['start_ticks']==ticks
                    and ancestors_include(pid,base_pid,table) for base_pid,ticks in self.baseline):
                continue
            inherited=False
            try:
                inherited=(NONCE_ENV+'='+self.nonce).encode() in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            except OSError:pass
            ancestor=root_live and ancestors_include(pid,self.root,table)
            adopted=row['ppid']==self.parent and pid!=self.root
            if key in self.known or key==self.root_key or ancestor or inherited or adopted:
                self.known[key]=dict(row,nonce_observed=inherited,live_root_ancestry=ancestor,
                    adopted_by_subreaper=adopted)
                found.append(self.known[key])
        return found

    def reap(self):
        for row in self.observe():
            if row['pid']==self.root or row['state']!='Z' or row['ppid']!=self.parent:continue
            try:
                pid,status=os.waitpid(row['pid'],os.WNOHANG)
                if pid:self.reaped.append(dict(pid=pid,start_ticks=row['start_ticks'],wait_status=status))
            except ChildProcessError:continue

    def live(self):
        self.reap()
        return [r for r in self.observe() if r['state']!='Z']

    def finish(self,process,grace_seconds,directory):
        """Kill only exact owned PID/start-time identities and retain lock until exit."""
        initial=self.live();events=[]
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGKILL):
            live=self.live()
            if not live:break
            for row in live:
                try:
                    current=process_table().get(row['pid'])
                    if current is None or current['start_ticks']!=row['start_ticks']:continue
                    os.kill(row['pid'],sig)
                    events.append(dict(signal=sig.name,pid=row['pid'],start_ticks=row['start_ticks'],
                        process_group=row['process_group'],session=row['session']))
                except ProcessLookupError:continue
            deadline=time.monotonic()+(.5 if sig==signal.SIGKILL else grace_seconds)
            while self.live() and time.monotonic()<deadline:
                process.poll();time.sleep(.025)
        # A D-state owner cannot justify releasing the GPU lock. Like the
        # original group helper, keep waiting and killing through interrupts.
        while self.live():
            try:
                for row in self.live():
                    try:os.kill(row['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
                time.sleep(.05);process.poll()
            except KeyboardInterrupt:continue
        self.reap()
        return dict(initial_live_owned=initial,events=events,live_owned=[],
            adopted_children_reaped=self.reaped,observed_identities=list(self.known.values()),
            scope='exact owned PID/start_ticks across process groups and sessions')
