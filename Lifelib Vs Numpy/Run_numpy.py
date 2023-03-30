import pandas as pd
import time
import numpy as np
import os
path = os.path.abspath(os.getcwd())

start_time = time.time()

i = 0.015

mt = pd.read_excel(path +'/fastlife/model/Input/MortalityTables.xlsx', header=1).iloc[1:]

pol = pd.read_excel(path +'/fastlife/model/Input/PoliyData.xlsx')



print("numpy approach run in %.2f seconds with 10K policies" % (time.time() - start_time))