from enum import Enum
from pathlib import Path


class Models(Enum):
    """
    Selection of Model
    """
    KINOVA_DEPRECATED = Path(__file__).parent.__str__() + "/KINOVA_arm_deprecated.bioMod"

    KINOVA_RIGHT = Path(__file__).parent.__str__() + "/KINOVA_arm_right.bioMod"
    KINOVA_RIGHT_SLIDE = Path(__file__).parent.__str__() + "/KINOVA_arm_right_plus_slide.bioMod"
    KINOVA_LEFT_REVERSED = Path(__file__).parent.__str__() + "/KINOVA_arm_reverse_left.bioMod"
    KINOVA_RIGHT_REVERSED = Path(__file__).parent.__str__() + "/KINOVA_arm_reverse_right.bioMod"

    WU = Path(__file__).parent.__str__() + "/wu_converted_definitif.bioMod"

    WU_INVERSE_KINEMATICS = Path(__file__).parent.__str__() + "/wu_converted_definitif_inverse_kinematics.bioMod"
    WU_INVERSE_KINEMATICS_XYZ = Path(__file__).parent.__str__() + "/wu_converted_definitif_inverse_kinematics_XYZ.bioMod"
    WU_INVERSE_KINEMATICS_XYZ_OFFSET = Path(__file__).parent.__str__() + "/wu_converted_definitif_inverse_kinematics_XYZ_offset.bioMod"

    WU_WITHOUT_FLOATING_BASE = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base.bioMod"

    WU_WITHOUT_FLOATING_BASE_TEMPLATE = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template.bioMod"
    WU_WITHOUT_FLOATING_BASE_VARIABLES = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_with_variables.bioMod"

    WU_WITHOUT_FLOATING_BASE_FIXED_TEMPLATE = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_fixed.bioMod"
    WU_WITHOUT_FLOATING_BASE_FIXED_VARIABLES = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_fixed_with_variables.bioMod"

    WU_WITHOUT_FLOATING_BASE_OFFSET_TEMPLATE = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_xyz_offset.bioMod"
    WU_WITHOUT_FLOATING_BASE_OFFSET_VARIABLES = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_xyz_offset_with_variables.bioMod"

    WU_WITHOUT_FLOATING_BASE_QUAT_TEMPLATE = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_quat.bioMod"
    WU_WITHOUT_FLOATING_BASE_QUAT_VARIABLES = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_quat_with_variables.bioMod"

    WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_TEMPLATE = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_quat_degroote.bioMod"
    WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_VARIABLES = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_quat_with_variables_degroote.bioMod"

    WU_WITHOUT_MODIF = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_modif.bioMod"
    WU_WITHOUT_MODIF_QUAT = Path(__file__).parent.__str__() + "/wu_converted_definitif_without_modif_quat.bioMod"

    WU_AND_KINOVA = Path(__file__).parent.__str__() + "/KINOVA_merge.bioMod"
    WU_AND_KINOVA_INVERSE_KINEMATICS = Path(__file__).parent.__str__() + "/KINOVA_merge_inverse_kinematics.bioMod"

    WU_AND_KINOVA_WITHOUT_FLOATING_BASE_TEMPLATE = Path(__file__).parent.__str__() + "/KINOVA_merge_without_floating_base_template.bioMod"
    WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_VARIABLES = Path(__file__).parent.__str__() + "/KINOVA_merge_without_floating_base_template_with_variables.bioMod"

    WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_TEMPLATE = Path(__file__).parent.__str__() + "/KINOVA_merge_without_floating_base_with_6_dof_support_template.bioMod"
    WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES = Path(__file__).parent.__str__() + "/KINOVA_merge_without_floating_base_with_6_dof_support_template_with_variables.bioMod"

    WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_TEMPLATE = Path(__file__).parent.__str__() + "/KINOVA_merge_without_floating_base_with_rototrans_template.bioMod"
    WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_VARIABLES = Path(__file__).parent.__str__() + "/KINOVA_merge_without_floating_base_with_rototrans_template_with_variables.bioMod"

