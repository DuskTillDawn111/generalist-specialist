"""Resource accounting for Roomie-style overlap, without performance counters.

Standalone active blocks are supplied by NVIDIA's installed occupancy calculator.
Spatial aggregate resource accounting below is a disclosed average-SM approximation.
"""
import math
from dataclasses import dataclass
@dataclass(frozen=True)
class DeviceLimits:
 warps: int
 blocks: int
 registers: int
 shared_bytes: int
 sm_count: int
 reserved_shared_per_block: int=1024

def resource_vector(k,limits):
 w=math.ceil(k['threads_per_block']/32)
 # Ada register allocation is per warp in units of 256 32-bit registers.
 regs=math.ceil(k['registers_per_thread']*32/256)*256*w
 shared=math.ceil((k['static_shared_bytes']+k['dynamic_shared_bytes'])/256)*256+limits.reserved_shared_per_block
 return (w,1,regs,shared)

def active_grid_blocks(k,limits):
 return min(k['max_active_blocks_per_sm'],k['grid_blocks']/limits.sm_count)

def can_coexist(a,b,limits):
 va=resource_vector(a,limits);vb=resource_vector(b,limits)
 ba=active_grid_blocks(a,limits);bb=active_grid_blocks(b,limits)
 return all(x*ba+y*bb<=cap+1e-12 for x,y,cap in zip(va,vb,(limits.warps,limits.blocks,limits.registers,limits.shared_bytes)))

def adjusted_blocks(k,remaining,limits):
 v=resource_vector(k,limits)
 b=min([active_grid_blocks(k,limits)]+[math.floor(cap/demand) for cap,demand in zip(remaining,v) if demand])
 return max(0,b)

def adjusted_rates(a,b,limits,policy='EQUAL_PARTITION'):
 if policy=='FCFS_CONTEXT_SERIAL':return (0.,1.)
 if can_coexist(a,b,limits):return (1.,1.)
 if policy!='EQUAL_PARTITION':raise ValueError(policy)
 capacities=(limits.warps,limits.blocks,limits.registers,limits.shared_bytes)
 half=tuple(c/2 for c in capacities)
 ba=adjusted_blocks(a,half,limits);bb=adjusted_blocks(b,half,limits)
 # If an indivisible launch cannot fit the share, preserve progress by serialization.
 if ba<=0 or bb<=0:return (0.,1.)
 return (min(1.,ba/active_grid_blocks(a,limits)),min(1.,bb/active_grid_blocks(b,limits)))
