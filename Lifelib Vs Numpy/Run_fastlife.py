import pandas as pd
import modelx as mx
import time

start_time = time.time()

proj = mx.read_model("fastlife/model").Projection
pols = proj.Policy.PolicyData()

pv_prem = proj.PV_PremIncome(0)
pv_claim = proj.PV_BenefitTotal(0)

for t in range(1,pols['PolicyTerm'].max()+1):
    pv_prem = pd.concat([pv_prem,proj.PV_PremIncome(t)],axis=1)
    pv_claim = pd.concat([pv_claim, proj.PV_BenefitTotal(t)], axis=1)

pv_net_cash_flow = pv_prem.to_numpy() + pv_claim.to_numpy()


print("fastlife run in %.2f seconds with 10K policies" % (time.time() - start_time))