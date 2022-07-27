from enum import Enum
from pathlib import Path


class Models(Enum):
    """
    Selection of Model
    """

    KINOVA_DEPRECATED = str(Path("KINOVA_arm_deprecated.bioMod").absolute())
    KINOVA_LEFT = str(Path("KINOVA_arm_reverse_left.bioMod").absolute())
    KINOVA_RIGHT = str(Path("KINOVA_arm_reverse_right.bioMod").absolute())
    KINOVA_MERGE = str(Path("KINOVA_merge.bioMod").absolute())
    KINOVA_MERGE_INVERSE_KINEMATICS = str(Path("KINOVA_merge_inverse_kinematics.bioMod").absolute())
    KINOVA_MERGE_WITHOUT_FLOATING_BASE_TEMPLATE = str(Path("KINOVA_merge_without_floating_base_template.bioMod").absolute())
    KINOVA_MERGE_WITHOUT_FLOATING_BASE_WITH_VARIABLES = str(Path("KINOVA_merge_without_floating_base_template_with_variables.bioMod").absolute())
    KINOVA_MERGE_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_TEMPLATE = str(Path("KINOVA_merge_without_floating_base_with_6_dof_support_template.bioMod").absolute())
    KINOVA_MERGE_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES = str(Path("KINOVA_merge_without_floating_base_with_6_dof_support_template_with_variables.bioMod").absolute())
    KINOVA_MERGE_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_TEMPLATE = str(Path("KINOVA_merge_without_floating_base_with_rototrans_template.bioMod").absolute())
    KINOVA_MERGE_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_VARIABLES = str(Path("KINOVA_merge_without_floating_base_with_rototrans_template_with_variables.bioMod").absolute())

    WU = str(Path("wu_converted_definitif.bioMod").absolute())
    WU_INVERSE_KINEMATICS = str(Path("wu_converted_definitif_inverse_kinematics.bioMod").absolute())
    WU_INVERSE_KINEMATICS_XYZ = str(Path("wu_converted_definitif_inverse_kinematics_XYZ.bioMod").absolute())
    WU_WITHOUT_FLOATING_BASE = str(Path("wu_converted_definitif_without_floating_base.bioMod").absolute())
    WU_WITHOUT_FLOATING_BASE_TEMPLATE = str(Path("wu_converted_definitif_without_floating_base_template.bioMod").absolute())
    WU_WITHOUT_FLOATING_BASE_TEMPLATE_FIXED = str(Path("wu_converted_definitif_without_floating_base_template_fixed.bioMod").absolute())
    WU_WITHOUT_FLOATING_BASE_TEMPLATE_QUAT = str(Path("wu_converted_definitif_without_floating_base_template_quat.bioMod").absolute())
    WU_WITHOUT_FLOATING_BASE_VARIABLES = str(Path("wu_converted_definitif_without_floating_base_template_with_variables.bioMod").absolute())
    WU_WITHOUT_MODIF = str(Path("wu_converted_definitif_without_modif.bioMod").absolute())
    WU_WITHOUT_MODIF_QUAT = str(Path("wu_converted_definitif_without_modif_quat.bioMod").absolute())

