"""Explicit protocol timing recurrence, never identified CPU service demand.

Each worker has one FIFO reader. A shared serial collector has dequeue and
processing timing slacks. Reader arrival order at the collector is an explicit
model assumption; original enqueue sequence and unrelated parent work are not
fully observed. No legacy notification delays are clamped or overwritten.
"""
from fractions import Fraction

FIELDS={'reader_slack_ns','collector_dequeue_slack_ns','collector_processing_slack_ns'}


def checked(values):
    if (type(values) is not dict or set(values)!=FIELDS
            or any(type(v) not in (int,Fraction) or v<0 for v in values.values())):
        raise ValueError('Three explicit nonnegative protocol timing slacks required')
    return {k:Fraction(v) for k,v in values.items()}


class TimingStations:
    def __init__(self,start_ns=0):
        if type(start_ns) not in (int,Fraction) or start_ns<0:raise ValueError('Exact start clock required')
        self.start=Fraction(start_ns);self.reader_free={};self.collector_free=self.start
        self.last_collector_arrival=self.start

    def issue(self,occurrence_ns,producer,slacks):
        slacks=checked(slacks)
        if not producer or type(occurrence_ns) not in (int,Fraction) or occurrence_ns<self.start:
            raise ValueError('Explicit producer and occurrence clock required')
        at=max(Fraction(occurrence_ns),self.reader_free.get(producer,self.start))+slacks['reader_slack_ns']
        self.reader_free[producer]=at
        return at

    def accept_reader(self,reader_ns,slacks):
        slacks=checked(slacks);reader_ns=Fraction(reader_ns)
        if reader_ns<self.last_collector_arrival:
            raise ValueError('Shared collector must consume actual reader-availability order')
        dequeue=max(reader_ns,self.collector_free)+slacks['collector_dequeue_slack_ns']
        processing=dequeue+slacks['collector_processing_slack_ns']
        self.collector_free=processing;self.last_collector_arrival=reader_ns
        return dequeue,processing


def derive(messages):
    """Keep all measured START/PHASE/RESULT/READY; reject unsupported order.

    Each row has identity, producer, occurrence_ns, reader_ns, dequeue_ns,
    processing_ns. A unique identity is provenance only, not a generation key.
    """
    readers={};rows=[];ids=set()
    for row in sorted(messages,key=lambda r:(r['occurrence_ns'],str(r['identity']))):
        if row['identity'] in ids:raise ValueError('Distinct original protocol-message identities required')
        ids.add(row['identity'])
        stamps=[row[k] for k in ('occurrence_ns','reader_ns','dequeue_ns','processing_ns')]
        if any(type(v) is not int or v<0 for v in stamps) or stamps!=sorted(stamps):
            raise ValueError('Complete monotone native notification clocks required')
        previous=readers.get(row['producer'],0)
        reader_slack=row['reader_ns']-max(row['occurrence_ns'],previous)
        if reader_slack<0:raise ValueError('Observed producer reader order is not supported')
        readers[row['producer']]=row['reader_ns'];rows.append(dict(row,reader_slack_ns=reader_slack))
    collector=0;result=[]
    for row in sorted(rows,key=lambda r:(r['reader_ns'],str(r['identity']))):
        slacks=dict(reader_slack_ns=row['reader_slack_ns'],
            collector_dequeue_slack_ns=row['dequeue_ns']-max(row['reader_ns'],collector),
            collector_processing_slack_ns=row['processing_ns']-row['dequeue_ns'])
        checked(slacks);collector=row['processing_ns'];result.append(dict(row,protocol_slacks=slacks))
    # Exact reconstruction uses two event orders, never producer occurrence
    # order as an invented global collector arrival order.
    stations=TimingStations();readers=[]
    for row in sorted(result,key=lambda r:(r['occurrence_ns'],str(r['identity']))):
        reader=stations.issue(row['occurrence_ns'],row['producer'],row['protocol_slacks'])
        if reader!=row['reader_ns']:raise ValueError('Reader recurrence does not reconstruct source')
        readers.append((reader,str(row['identity']),row))
    for reader,_,row in sorted(readers):
        dequeue,processing=stations.accept_reader(reader,row['protocol_slacks'])
        if (dequeue,processing)!=(row['dequeue_ns'],row['processing_ns']):
            raise ValueError('Shared collector recurrence does not reconstruct source')
    return result
