from enum import Enum
from pathlib import Path


class Tasks(Enum):
    """
    Selection of tasks
    """

    ARMPIT = str(Path("F0_aisselle_05.c3d").absolute())
    DRINK = str(Path("F0_boire_05.c3d").absolute())
    TEETH = str(Path("F0_dents_05.c3d").absolute())
    DRAW = str(Path("F0_dessiner_05.c3d").absolute())
    EAT = str(Path("F0_manger_05.c3d").absolute())
    HEAD = str(Path("F0_tete_05.c3d").absolute())


class TasksKinova(Enum):
    """
    Selection of tasks
    """

    ARMPIT = str(Path("F3_aisselle_01.c3d").absolute())
    DRINK = str(Path("F3_boire_01.c3d").absolute())
    TEETH = str(Path("F3_dents_01.c3d").absolute())
    DRAW = str(Path("F3_dessiner_02.c3d").absolute())
    HEAD = str(Path("F3_tete_01.c3d").absolute())



