"""Inherit the existing frozen runtime environment before the GPU driver."""
import os
from .common import read,PROTOCOL,OUT

def main():
    if (OUT/'campaign_status.json').exists() and read(OUT/'campaign_status.json')['status']=='COMPLETE_AND_STOPPED':
        raise SystemExit('9BC is complete and frozen; do not relaunch GPU validation')
    for key,value in read(PROTOCOL)['execution_environment_overrides'].items():
        os.environ[key]=value
    from .gpu import main as execute
    execute()

if __name__=='__main__':main()
