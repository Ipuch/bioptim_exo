from enum import Enum
from pathlib import Path


class Tasks(Enum):
    """
    Selection of tasks
    """

    ARMPIT = Path(__file__).parent.__str__() + "/F0_aisselle_05.c3d"
    DRINK = Path(__file__).parent.__str__() + "/F0_boire_05.c3d"
    TEETH = Path(__file__).parent.__str__() + "/F0_dents_05.c3d"
    DRAW = Path(__file__).parent.__str__() + "/F0_dessiner_05.c3d"
    EAT = Path(__file__).parent.__str__() + "/F0_manger_05.c3d"
    HEAD = Path(__file__).parent.__str__() + "/F0_tete_05.c3d"


class TasksKinova(Enum):
    """
    Selection of tasks
    """

    ARMPIT = Path(__file__).parent.__str__() + "/F3_aisselle_01.c3d"
    DRINK = Path(__file__).parent.__str__() + "/F3_boire_01.c3d"
    TEETH = Path(__file__).parent.__str__() + "/F3_dents_01.c3d"
    DRAW = Path(__file__).parent.__str__() + "/F3_dessiner_02.c3d"
    HEAD = Path(__file__).parent.__str__() + "/F3_tete_01.c3d"
    HEAD_debug = Path(__file__).parent.__str__() + "/tete_debug.c3d"
    Drink_debug_45f = Path(__file__).parent.__str__() + "/boire_45f.c3d"
    Drink_debug_100f = Path(__file__).parent.__str__() + "/boire_100f.c3d"


