import pandas as pd
import time
import numpy as np
import os
path = os.path.abspath(os.getcwd())

start_time = time.time()

i = 0.015

mt = pd.read_excel(path +'/fastlife/model/Input/MortalityTables.xlsx', header=1).iloc[1:]
pol = pd.read_excel(path +'/fastlife/model/Input/PoliyData.xlsx')

# Create a zero and one numpy array of n_max and number_policy length
n_max = pol['PolicyTerm'].max() + 1
zeros = np.zeros((len(pol), n_max))
ones = np.ones((len(pol), n_max))

# Create age vector
age = ones.copy().cumsum(axis=1) + pol['IssueAge'].to_numpy()[:, np.newaxis] - 1
# Do the vectorized Qx
mt = mt.rename(columns={"Unnamed: 0": "age"}).set_index('age').T
qx_table = pol[['Sex']].merge(mt,how='left',left_on='Sex',right_index=True).to_numpy()[:,1:]
qx = np.take_along_axis(qx_table,age.astype(int),axis=1)


duration = ones.copy().cumsum(axis=1) -1
policy_term = ones.copy() * (pol['PolicyTerm']).to_numpy()[:, np.newaxis]
policy_maturity = zeros.copy()
pols_if_start = zeros.copy()
pols_death = zeros.copy()
pols_if_end = zeros.copy()
pv_prem = zeros.copy()
pv_claim = zeros.copy()
pols_if_end[:, 0] = 1
pols_death[:, 0] = qx[:, 0]

# Inforces
for t in range(1, n_max):
    pols_if_start[:, t] = pols_if_end[:, t - 1] - pols_death[:, t - 1]
    policy_maturity[:, t] = (policy_term[:, t] == duration[:, t]) * pols_if_start[:, t]
    pols_if_end[:, t] = pols_if_start[:, t] - policy_maturity[:, t]
    pols_death[:, t] = pols_if_end[:, t] * qx[:, t]

prem_income = ones.copy() * pol['Premium'].to_numpy()[:, np.newaxis] * pols_if_end
death_benefit = ones.copy() * pol['SumAssured'].to_numpy()[:, np.newaxis] * pols_death * (-1)

for t in range(0,n_max):
    discount_factor = 1 / (1+i)**(ones.copy()[:,t:].cumsum(axis=1)-1)
    pv_prem[:, t] = (prem_income[:, t:] * discount_factor).sum(axis=1)
    pv_claim[:, t] = (death_benefit[:, t:] * discount_factor / (1 + i)).sum(axis=1)

pv_net_cash_flow = pv_prem + pv_claim

print("numpy approach run in %.2f seconds with 10K policies" % (time.time() - start_time))