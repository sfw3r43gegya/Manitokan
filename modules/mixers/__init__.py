REGISTRY = {}

from modules.mixers.Q_mixer import QMixer
from modules.mixers.vdn_mixer import VDNMixer

from modules.mixers.Qtrans_mixer import QTranBase
from modules.mixers.MAVEN import MAVENmixer

REGISTRY["qmix"] = QMixer
REGISTRY["vdn"] = VDNMixer
REGISTRY["qtrans"] = QTranBase
REGISTRY["maven"] = MAVENmixer