
# coding: utf-8

# In[5]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


# In[32]:

Rs = [np.load('exp_r.np.npy'),np.load('exp3p_r.np.npy'),np.load('exp3ix_r.npy')]
fig,ax = plt.subplots(figsize=(15,15))
c = np.load('exp3ix_c.npy')

for R,label in zip(Rs,['EXP3','EXPR.P','EXP3-IX']):
    m, s = np.mean(R, axis=1), np.std(R, axis=1, ddof=1)*1.96/np.sqrt(50)
    ax.errorbar(c,m,s, label=label)
    ax.set_xticks(c)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel(r"$\frac{\eta}{\sqrt{\frac{2\log(K)}{KT}}}$", fontsize=40, labelpad=20)
    ax.set_ylabel(r"Expected regret", fontsize=20, labelpad=30)
    ax.set_title('Expected Pseudo Regret for various algorithms', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('Q2.png')
plt.show()

