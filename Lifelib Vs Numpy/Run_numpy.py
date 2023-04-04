import pandas as pd
import time
import numpy as np
import os

path = os.path.abspath(os.getcwd())

start_time = time.time()

i = 0.015

mt = pd.read_excel(path +'/fastlife/model/Input/MortalityTables.xlsx', header=1).iloc[1:]

pol = pd.read_excel(path +'/fastlife/model/Input/PoliyData.xlsx')



n_max = pol['MaxPolicyTerm'].max()
number_policy = len(pol)

# Create a zero and one numpy array of n_max and number_policy length
zeros = np.zeros((number_policy, n_max))
ones = np.ones((number_policy, n_max))


lapse = ones * 0.1

# Do the vectorized Qx
mt = mt.rename(columns={"Unnamed: 0": "age"}).set_index('age').T
qx_table = pol[['Sex']].merge(mt,how='left',left_on='Sex',right_index=True).to_numpy()[:,1:]


# Create age vector
age_end = ones * pol['MaxPolicyTerm'].to_numpy()[:, np.newaxis]
age = (ones).cumsum(axis=1) + pol['IssueAge'].to_numpy()[:, np.newaxis] - 1

# we max age at 130 (max mt)
age[age>=130] = (ones * 130)[age>=130]

# ccreate vector
qx = np.take_along_axis(qx_table,age.astype(int),axis=1)
qx[age > age_end] = 0

# todo à vérifier
duration = ones.cumsum(axis=1)

# Lapse rate
lapse_rate = np.maximum(0.1 - 0.02 * duration, ones * 0.02)

pols_maturity = zeros.copy()

pols_death = zeros.copy()
pols_if = zeros.copy()
pols_lapse = zeros.copy()
policy_term = ones * (pol['MaxPolicyTerm']).to_numpy()[:, np.newaxis]

pols_if[:, 0] = 1 # Same as pol_init

# Inforces
for i in range(0, n_max):
    pols_maturity[:, i] = pols_if[:, i - 1] * (policy_term[:, i] == duration[:, i])*1
    pols_death[:, i] = pols_if[:, i] * qx[:, i]
    pols_lapse[:, i] = (pols_if[:, i] - pols_death[:, i]) * (1 - (1 - lapse_rate[:,i])**(1/12))

    if i == 0:
        pols_if[:,i] = ones.copy()[:,i]
    else:
        pols_if[:, i] = pols_if[:, i - 1] - pols_death[:, i - 1] - pols_lapse[:, i - 1] - pols_maturity[:, i]




print("numpy approach run in %.2f seconds with 10K policies" % (time.time() - start_time))