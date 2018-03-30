
# coding: utf-8

# In[17]:

import numpy as np
import matplotlib.pyplot as plt

def exp3p(d,T,eta,beta=None,gamma=None):
    beta = eta
    gamma = eta*d
    p       = np.zeros([d,T])
    e_gain = 0
    elv = 0.5*np.ones([d,2])
    elv[-2,:] = 0.4
    elv[-1,:] = [0.6,0.3]
    egv = 1-elv
    G = np.zeros([d])
    w_tilde = np.ones([d])

    for t in range(T):
        p[:,t] = w        = (1-gamma)*(w_tilde/np.sum(w_tilde)) + gamma/d
        adv_choice        = np.random.choice(d,p=w)
        e_gain_c          = egv[adv_choice,(2*t)//T]
        gain              = beta/w
        gain[adv_choice] += np.random.choice(2,p=[1-e_gain_c, e_gain_c])/w[adv_choice]
        e_gain           += e_gain_c
        w_tilde           = w_tilde*np.exp(eta*gain)

    return 0.6*T - e_gain, p


# In[18]:

import numpy as np
import matplotlib.pyplot as plt

def exp3(d,T,eta):
    p       = np.zeros([d,T])
    e_loss = 0
    elv = 0.5*np.ones([d,2])
    elv[-2,:] = 0.4
    elv[-1,:] = [0.6,0.3]
    
    w_tilde = np.ones([d])
    
    for t in range(T):
        p[:,t] = w = w_tilde/np.sum(w_tilde)
        adv_choice = np.random.choice(d,p=w)
        e_loss_c   = elv[adv_choice,(2*t)//T]
        l          = np.random.choice(2,p=[1-e_loss_c, e_loss_c])/w[adv_choice]
        e_loss    += e_loss_c
        w_tilde[adv_choice]    = w_tilde[adv_choice]*np.exp(-eta*l)

    return e_loss - 0.4*T, p


# In[19]:

def exp3ix(d,T,eta,gamma=None):
    gamma = eta/2
    p       = np.zeros([d,T])
    e_loss = 0
    elv = 0.5*np.ones([d,2])
    elv[-2,:] = 0.4
    elv[-1,:] = [0.6,0.3]
    
    w_tilde = np.ones([d])

    for t in range(T):
        p[:,t] = w = w_tilde/np.sum(w_tilde)
        adv_choice = np.random.choice(d,p=w)
        e_loss_c   = elv[adv_choice,(2*t)//T]
        l          = np.random.choice(2,p=[1-e_loss_c, e_loss_c])/(w[adv_choice]+gamma)
        e_loss    += e_loss_c
        w_tilde[adv_choice]    = w_tilde[adv_choice]*np.exp(-eta*l)

    return e_loss - 0.4*T, p


# In[ ]:

d = 10      #Number of advisors
T = 100000  #Number of rounds
c = 1.0
eta = np.sqrt(2.0*np.log(d)/T)
fig, axs   = plt.subplots(1,3, figsize=(15,7))
for func,ax in zip([exp3,exp3p,exp3ix],axs):
    p_r, p    = func(d,T,c*eta)
    cf = ax.contourf(p, 200,vmin=0, vmax=1)
    ax.set_title(r'$\eta$ multilier =' + str(c), fontsize=25)
    ax.set_xlabel("Time", fontsize=20)
    ax.set_ylabel("Advisor", fontsize=20)
    #ax.set_xticks([50000,100000])
    ax.tick_params(axis='both', labelsize=15)

fig.tight_layout()
fig.subplots_adjust(bottom=0.4)
cbar_ax = fig.add_axes([0.05,0.24, 0.91, 0.04])
cb = fig.colorbar(cf,cax=cbar_ax, ticks=np.linspace(0,1,11), orientation='horizontal')
cb.set_label("Probability", fontsize=20)
cb.ax.tick_params(axis='both', labelsize=15)

plt.savefig('./Q3.png')
plt.show()


# In[ ]:



