import numpy as np
#import matplotlib.pyplot as plt

def wma(d,T,eta):
    w_tilde = 1.0*np.ones([d])
    l       = np.zeros([d,T])
    loss = 0
    e_loss = 0
    for t in range(T):
        w = w_tilde/np.sum(w_tilde)
        #adv_choice = np.random.choice(d,p=w)
        l[:-2,t] = np.random.choice(2, size=8, p=[0.5, 0.5])
        l[-2,t]  = np.random.choice(2, p=[0.6,0.4])
        delta    = 0.1 if t<T/2 else -0.2
        l[-1,t]  = np.random.choice(2, p=[0.5-delta,0.5+delta])
        #loss    += l[adv_choice,t]
        e_loss  += w.dot(l[:,t])
        w_tilde  = w_tilde*np.exp(-eta*l[:,t])

    costs    = np.sum(l,axis=1) 
    regret   = loss - np.min(costs)
    p_regret = e_loss - np.min(costs)
    
    return p_regret

if __name__ == '__main__':
    d = 10      #Number of advisors
    T = 100000  #Number of rounds

    c   = np.linspace(0.1,2.1,11)
    Eta = c*np.sqrt(2.0*np.log(d)/T)
    R   = []
    n_samples=30
    for eta in Eta:
        pr = 0
        for trial in range(n_samples):
            pr += 1.0*wma(d,T,eta)/n_samples
        R.append(pr)
    
    print(R)
    print(c)
#    fig,ax = plt.subplots(figsize=(20,20))
#    ax.plot(R,c)
#    ax.set_xticks(C)
#    ax.set_xlabel("$\frac{eta}{abs}$")
#    plt.savefig("q1.png")
#    plt.show()
