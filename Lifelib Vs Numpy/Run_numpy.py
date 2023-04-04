import pandas as pd
import time
import numpy as np
import os
path = os.path.abspath(os.getcwd())

start_time = time.time()

i = 0.015

mt = pd.read_excel(path +'/fastlife/model/Input/MortalityTables.xlsx', header=1).iloc[1:]

pol = pd.read_excel(path +'/fastlife/model/Input/PoliyData.xlsx')

n_max = pol['PolicyTerm'].max() + 1
number_policy = len(pol)

# Create a zero and one numpy array of n_max and number_policy length
zeros = np.zeros((number_policy, n_max))
ones = np.ones((number_policy, n_max))

# Create age vector
age = ones.copy().cumsum(axis=1) + pol['IssueAge'].to_numpy()[:, np.newaxis] - 1

# Do the vectorized Qx
mt = mt.rename(columns={"Unnamed: 0": "age"}).set_index('age').T
qx_table = pol[['Sex']].merge(mt,how='left',left_on='Sex',right_index=True).to_numpy()[:,1:]
qx = np.take_along_axis(qx_table,age.astype(int),axis=1)


duration = ones.copy().cumsum(axis=1) -1
policy_term = ones.copy() * (pol['PolicyTerm']).to_numpy()[:, np.newaxis]
PolsMaturity = zeros.copy()
PolsIF_End = zeros.copy()
PolsDeath = zeros.copy()
PolsIF_AftMat = zeros.copy()

PolsIF_AftMat[:,0]=1
PolsDeath[:,0] = qx[:, 0]

# Inforces
for t in range(1, n_max):
    PolsIF_End[:, t] = PolsIF_AftMat[:, t - 1] - PolsDeath[:, t - 1]
    PolsMaturity[:, t] = (policy_term[:, t] == duration[:, t]) * PolsIF_End[:, t]
    PolsIF_AftMat[:, t] = PolsIF_End[:, t] - PolsMaturity[:, t]
    PolsDeath[:, t] = PolsIF_AftMat[:, t] * qx[:, t]


PremInc = ones.copy() * pol['Premium'].to_numpy()[:, np.newaxis] * PolsIF_AftMat
BenefitDeath = ones.copy() * pol['SumAssured'].to_numpy()[:, np.newaxis] * PolsDeath * (-1)

pv_prem = zeros.copy()
pv_claim = zeros.copy()


for t in range(0,n_max):
    discount_factor = 1 / (1+i)**(ones.copy()[:,t:].cumsum(axis=1)-1)
    pv_prem[:, t] = (PremInc[:,t:] * discount_factor).sum(axis=1)
    pv_claim[:, t] = (BenefitDeath[:, t:] * discount_factor/(1+i)).sum(axis=1)

pv_net = pv_prem + pv_claim


print("numpy approach run in %.2f seconds with 10K policies" % (time.time() - start_time))