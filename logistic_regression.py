# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:12:22 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

loansData = pd.read_csv('loansData_clean.csv')

loansData['IR_TF'] = map (lambda x: 1 if x >= .12 else 0, loansData['clean_Interest_rate'])

loansData['intercept'] = map (lambda x: 1, loansData['clean_Interest_rate'])

loanamt = loansData['Amount.Requested']
fico = loansData['FICO_Score']

ind_vars = ['intercept', 'Amount.Requested', 'FICO_Score']

logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])
result = logit.fit()

coeff = result.params
print(coeff)

def logistic_function(x,y):
    return float (1/ (1 + math.exp(coeff.get('intercept') + coeff.get('Amount.Requested') * x + coeff.get('FICO_Score') * y)))

print("p = %f" % logistic_function(1000,720))
if logistic_function(1000,720)>= .70:
    print("Yes!")
else:
    print("Maybe some other time" )

#Make a variable with the predicted proabilities

loansData['p_hat'] = map (logistic_function, loansData['Amount.Requested'], loansData['FICO_Score'])

# Here I plot FICO vs predicted probabilities

plt.plot(loansData['FICO_Score'],loansData['p_hat'], ".")
plt.show()