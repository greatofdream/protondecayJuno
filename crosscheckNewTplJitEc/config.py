'''
the threshold used to select signal accroding to number of michel electron
'''
import numpy as np
# nMichel=0, 1, 2
thresholds = np.zeros((3,2), dtype=[
    ('eR_l', '<f4'), ('eR_r', '<f4'),
    ('Qedep_l', '<f4'), ('Qedep_r', '<f4'),
    ('nCap_l', '<f4'), ('nCap_r', '<f4'),
    ('michelDist_l', '<f4'), ('michelDist_r', '<f4'),
    ('npeak_l', '<f4'), ('npeak_r', '<f4'),
    ('t12_l', '<f4'), ('t12_r', '<f4'),
    ('E2_1_l', '<f4'), ('E2_1_r', '<f4'),
    ('E2_2_l', '<f4'), ('E2_2_r', '<f4'),
    ('E1_l', '<f4'), ('E1_r', '<f4'),
    ('chi12_l', '<f4'), ('chi2_r', '<f4'),
    ('neutronDist_l', '<f4'), ('neutronDist_r', '<f4'),
])
# using 40000 as distance unbound; 1000 as number unbound
# nmichel = 0
thresholds[0, 0] = (
    0, 17500,
    200, 600,
    0, 0,
    0, 17000,
    1, 10,
    12.5, 1000,
    85, 230,
    325, 430,
    20, 240,
    3.5, 1000,
    0, 40000
)
# nmichel = 1, nCapture = 0
thresholds[1, 0] = (
    0, 17500,
    200, 600,
    0, 0,
    0, 800,
    1, 10,
    6, 1000,
    85, 230,
    230, 430,
    20, 230,
    1.11, 1000,
    0, 40000
)
# nmichel = 1, nCapture >= 1
thresholds[1, 1] = (
    0, 17500,
    200, 600,
    1, 1000,
    0, 700,
    1, 10,
    7.8, 1000,
    85, 230,
    230, 430,
    20, 230,
    1, 1000,
    0, 750
)
# nmichel = 2
thresholds[2, 0] = (
    0, 17500,
    200, 600,
    0, 1000,
    0, 800,
    1, 10,
    5.1, 1000,
    85, 230,
    230, 430,
    20, 230,
    1.1, 1000,
    0, 40000
)