"""Repeated polling snapshots must preserve the exact serial owner identity."""
import copy
import sqlite3
import unittest

from tools.predictor_validation_20261027 import whole57_lifecycle_training_adapter_v35 as v


class StartupSnapshotTests(unittest.TestCase):
    def fixture(self):
        owners=list(v.RESIDENTS)
        identities={o:dict(pid=100+i,start_ticks=200+i,execution_nonce='nonce') for i,o in enumerate(owners)}
        snapshots=[dict(phase='pre_startup',time_ns=0,launchers={},ready={},nvml_processes_mib={})]
        for i in range(4):
            snapshots.append(dict(phase='startup_begin',time_ns=10+10*i,
                launchers={o:copy.deepcopy(identities[o]) for o in owners[:i+1]},ready={o:{} for o in owners[:i]}))
        repeated=copy.deepcopy(snapshots[1]);repeated['time_ns']=11;snapshots.insert(2,repeated)
        snapshots.append(dict(phase='all_ready',time_ns=50,launchers=copy.deepcopy(identities),ready={o:{} for o in owners}))
        raw=dict(ready={o:dict(pid=i['pid']) for o,i in identities.items()},live_identity_events=snapshots,
            live_owned_processes={o:dict(start_ticks=i['start_ticks']) for o,i in identities.items()})
        events={(1,0,10+i,1000+i):[dict(timestamp=15+10*i)] for i in range(4)}
        events[(1,0,99,1001)]=[dict(timestamp=70)]
        connection=sqlite3.connect(':memory:');self.addCleanup(connection.close)
        connection.execute('CREATE TABLE TARGET_INFO_SESSION_START_TIME(systemClockNs INTEGER)')
        connection.execute('INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES(0)')
        return connection,raw,dict(execution_nonce='nonce'),events,{}

    def test_unchanged_poll_repetition_retains_four_windows_and_extra_context_identity(self):
        connection,raw,context,events,brackets=self.fixture()
        mapping,evidence=v.context_mapping(connection,raw,context,events,brackets)
        self.assertEqual(len(evidence['primary_creation_windows']),4)
        self.assertEqual(mapping[(1,0,99,1001)],list(v.RESIDENTS)[1])
        self.assertFalse(evidence['arithmetic_PID_join'])
        self.assertFalse(evidence['driver_and_CUPTI_context_ID_equality_assumed'])

    def test_repeated_poll_cannot_change_PID_start_ticks_nonce_or_READY(self):
        for key,value in [('pid',999),('start_ticks',999),('execution_nonce','changed'),('ready',{'unexpected':{}})]:
            connection,raw,context,events,brackets=self.fixture()
            repeated=raw['live_identity_events'][2]
            if key=='ready':repeated[key]=value
            else:repeated['launchers'][list(v.RESIDENTS)[0]][key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):
                v.context_mapping(connection,raw,context,events,brackets)


if __name__=='__main__':unittest.main()
