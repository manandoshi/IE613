
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

def wma(d,T,eta):
    w_tilde = np.ones([d])
    l       = np.zeros([d,T])
    p       = np.zeros([d,T])
    #loss    = 0
    e_loss  = 0
    for t in range(T):
        p[:,t] = w = w_tilde/np.sum(w_tilde)
        #adv_choice = np.random.choice(d,p=w)
        l[:-2,t]   = np.random.choice(2, size=8, p=[0.5, 0.5])
        l[-2,t]    = np.random.choice(2, p=[0.6,0.4])
        delta      = 0.1 if t<T/2 else -0.2
        l[-1,t]    = np.random.choice(2, p=[0.5-delta,0.5+delta])
        #loss      += l[adv_choice,t]
        e_loss    += w.dot(l[:,t])
        w_tilde    = w_tilde*np.exp(-eta*l[:,t])

    costs    = np.sum(l,axis=1) 
    #regret   = loss - np.min(costs)
    p_regret = e_loss - np.min(costs)
    
    return p_regret, p


# In[ ]:

d = 10      #Number of advisors
T = 100000  #Number of rounds
C = np.array([0.1,0.5,2.1])
eta       = np.sqrt(2.0*np.log(d)/T)
fig, axs   = plt.subplots(1,3, figsize=(15,7))
for c,ax in zip(C,axs):
    p_r, p    = wma(d,T,c*eta)
    cf = ax.contourf(p, 200,vmin=0, vmax=1)
    ax.set_title(r'$\eta$ multilier =' + str(c), fontsize=25)
    ax.set_xlabel("Time", fontsize=20)
    ax.set_ylabel("Advisor", fontsize=20)
    ax.set_xticks([50000,100000])
    ax.tick_params(axis='both', labelsize=15)

fig.tight_layout()
fig.subplots_adjust(bottom=0.4)
cbar_ax = fig.add_axes([0.05,0.24, 0.91, 0.04])
cb = fig.colorbar(cf,cax=cbar_ax, ticks=np.linspace(0,1,11), orientation='horizontal')
cb.set_label("Probability", fontsize=20)
cb.ax.tick_params(axis='both', labelsize=15)

plt.savefig('./q1b_b.png')
plt.show()


# In[ ]:

d = 10      #Number of advisors
T = 100000  #Number of rounds

c         = np.linspace(0.1,2.1,11)
Eta       = c*np.sqrt(2.0*np.log(d)/T)
n_samples = 30
R         = np.zeros([11,n_samples])
for i,eta in enumerate(Eta):
    for trial in range(n_samples):
        R[i,trial],_ = wma(d,T,eta)
        print("Sample: {}, i_c:{}".format(trial,i))


# In[ ]:

m, s   = np.mean(R, axis=1), np.std(R, axis=1, ddof=1)*1.96/np.sqrt(n_samples)

fig,ax = plt.subplots(figsize=(15,15))
ax.errorbar(c,m,s)
ax.set_xticks(c)
ax.tick_params(axis='both', labelsize=15)
ax.set_xlabel(r"$\frac{\eta}{\sqrt{\frac{2\log(d)}{T}}}$", fontsize=40, labelpad=20)
ax.set_ylabel(r"Expected regret", fontsize=20, labelpad=30)
plt.savefig(r"./plots/q1.png")
plt.show()

