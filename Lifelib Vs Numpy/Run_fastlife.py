import numpy as np
import pandas as pd
import modelx as mx
import time

start_time = time.time()

proj = mx.read_model("fastlife/model").Projection
pols = proj.Policy.PolicyData()

pv_prem = proj.PV_PremIncome(0)
pv_claim = proj.PV_BenefitTotal(0)
pv_net = proj.PV_NetCashflow(0)
premInc = proj.PremIncome(0)
benefitTotal = proj.BenefitTotal(0)

for t in range(1,pols['PolicyTerm'].max()+1):

    pv_prem = pd.concat([pv_prem,proj.PV_PremIncome(t)],axis=1)
    pv_claim = pd.concat([pv_claim, proj.PV_BenefitTotal(t)], axis=1)
    pv_net = pd.concat([pv_net, proj.PV_NetCashflow(t)], axis=1)
    premInc = pd.concat([premInc,proj.PremIncome(t)],axis=1)
    benefitTotal = pd.concat([benefitTotal,proj.BenefitTotal(t)],axis=1)

NetCashflow = pv_prem.to_numpy() + pv_claim.to_numpy()


print("fastlife run in %.2f seconds with 10K policies" % (time.time() - start_time))