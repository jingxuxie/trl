from agents.trl import TRLAgent
from agents.bmm_trl import BMMTRLAgent
from agents.crl import CRLAgent
from agents.qrl import QRLAgent
from agents.gcfbc import GCFBCAgent
from agents.gciql import GCIQLAgent

agents = dict(
    trl=TRLAgent,
    bmm_trl=BMMTRLAgent,
    crl=CRLAgent,
    qrl=QRLAgent,
    gcfbc=GCFBCAgent,
    gciql=GCIQLAgent,
)
