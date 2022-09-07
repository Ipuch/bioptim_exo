from lib import from_bo_to_pickle

k = ['/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_quat/tete/F0_tete_05_2022_08_25_17_18_04_468630.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/muscle_driven/aisselle/F0_aisselle_05_2022_08_26_10_01_36_700173.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/muscle_driven/dents/F0_dents_05_2022_08_17_19_38_12_717853.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/muscle_driven/manger/F0_manger_05_2022_08_23_19_26_42_078272.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/muscle_driven/tete/F0_tete_05_2022_08_24_10_55_31_451971.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_offset/aisselle/F0_aisselle_05_2022_08_26_14_05_20_110908.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_offset/boire/F0_boire_05_2022_08_26_13_55_55_996870.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_offset/dents/F0_dents_05_2022_08_26_13_53_24_362296.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_offset/manger/F0_manger_05_2022_08_26_14_48_55_312670.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_offset/tete/F0_tete_05_2022_08_26_13_58_21_171507.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_quat/aisselle/F0_aisselle_05_2022_08_25_16_14_00_581409.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_quat/boire/F0_boire_05_2022_08_25_17_13_18_773946.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_quat/dents/F0_dents_05_2022_08_25_16_58_37_810470.bo',
     '/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/event_tracking/solutions/torque_driven_quat/manger/F0_manger_05_2022_08_25_18_01_43_130921.bo',
     ]

for x in range(0,len(k)):
    from_bo_to_pickle(k[x])