import numpy as np
import xalglib
import adolc
import time
from scipy.integrate import odeint

def lorenz96(x, t):
    D = len(x)
    dxdt = []
    for i in range(D):
        dxdt.append(x[np.mod(i-1,D)]*(x[np.mod(i+1,D)]-x[np.mod(i-2,D)]) - x[i] + 8.17)
    
    return np.array(dxdt)


def gen_data(dt, T_final, skip=0):

    T_total = np.arange(0,T_final,dt)
    #data_initial = randn(5+1,1)
    #data_initial = [0.80, 0.95, 0.71, 0.24, 0.63,-1, 2, 1, -2.2, 0.3];
    #data_initial[-1] = 8.17
    data_initial = [ -2.33211 ,  -5.508487 , -9.208057, -10.98852 ,   6.165695]
    #[0.80, 0.95, 0.71, 0.24, 0.63];

    Y = odeint(lorenz96,data_initial,T_total)
    Y = Y[skip:skip+1000,:]
    param = 8.17*np.ones(len(Y))


    np.savetxt("data_D{0}_dt{1}_noP.txt".format(len(data_initial),dt),Y)
    noise = 0.5*np.random.standard_normal(Y.shape)
    Ynoise = Y + noise
    np.savetxt("dataN_D{0}_dt{1}_noP.txt".format(len(data_initial),dt),Ynoise)

    Y = np.column_stack((Y,param))
    np.savetxt("data_D{0}_dt{1}.txt".format(len(data_initial),dt),Y)
    noise = 0.5*np.random.standard_normal(Y.shape)
    Ynoise = Y + noise
    np.savetxt("dataN_D{0}_dt{1}.txt".format(len(data_initial),dt),Ynoise)

def array_map(x):
    f = np.zeros_like(x)
    for i in range(x.shape[0]):
        f[i,:] = map(x[i,:],model,dt,i*dt)
    return f

def action(x, beta):
#    x = np.array(x)
    x = np.reshape(x,(N,D))
    Rf = 0.01*(2**beta)
    Rm = 4.0
    Rtd = 1.0/(1.0/Rf+1.0/Rm)

    if x.shape != (N,D):
        raise "x is wrong dims!"

    # Rm term
    dy = x[:,measIdx]-y[:,:]
    #action = np.sum(0.5*Rm*(dy)**2)
    
    #Rf term
    f1 = array_map(x)
    dxf = 0.5*Rf*(x[1:,:]-f1[:-1,:])**2
    #action += np.sum(0.5*Rf*(x[1:,:]-f1[:-1,:])**2)
    
    #Rtd term
    
    dytd = np.zeros((Ntd,N,M),dtype=object)

    for count in range(Ntd):
        tau = taus[count]
        if count==0:
            for i in range(1,tau):
                f1 = array_map(f1)
        else:
            for i in range(taus[count-1],tau):
                f1 = array_map(f1)
        dytd[count,:-tau,:] = y[tau:,:]-f1[:-tau,measIdx]   
        #dxtd = x[tau:,measIdx]-f1[:-tau,measIdx]   
        
    action = np.zeros((N*M + (N-1)*D + np.sum((N-np.array(taus))*M)),dtype=object)
    action[:N*M] = dy.flatten()
    action[N*M: N*M+(N-1)*D] = dxf.flatten()
    last = N*M+(N-1)*D
    for count in range(Ntd):
        tau = taus[count]
        next = (N-tau)*M
        action[last:last+next] = dytd[count,:-tau,:].flatten()
        last += next
     #   action += np.sum(0.5*Rtd*(dytd)**2) + np.sum(0.5*Rf*(dxtd)**2)

    return action
   

def rk2(x,f,dt,t):
    k1 = dt*f(x,t)
    k2 = dt*f(x+0.5*k1,t+0.5*dt)
    return x + k2

def rk4(x,f,dt,t):
    k1 = dt*f(x,t)
    k2 = dt*f(x+0.5*k1, t+0.5*dt)
    k3 = dt*f(x+0.5*k2, t+0.5*dt)
    k4 = dt*f(x+k3, t+dt)
    return x + 1.0/6.0*(k1+2.0*k2+2.0*k3+k4)

def lm_func(x,f,p):
    f[:] = adolc.function(adolcID,x)
    return
def lm_jac(x,f,jac,p):
    tmpjac = adolc.jacobian(adolcID,x)    
    m = tmpjac.shape[0]
    n = tmpjac.shape[1]
    for i in range(m):
        jac[i][:] = tmpjac[i,:]

    f[:] = adolc.function(adolcID,x)
    return


def run(N,D,dt,beta,x0):
    
    if x0=='start':
        #init = 20.0*np.random.random_sample((N,D))-10.0
        #np.savetxt('initpaths.txt',init)        
        init = np.loadtxt('initpaths.txt')       
    else:
        init = x0.reshape(N,D)

    #init = np.loadtxt("data_D{0}_dt{1}_noP.txt".format(D,dt))[:N,:]
    
    if init.shape != (N,D):
        raise "x is wrong dims!"
  
    epsg = 1e-8
    epsf = 1e-8
    epsx = 1e-8
    maxits = 10000
    

    ### Use ADOL-C to generate trace, used for derivatives and
    ### evaluations
    start = time.time()

    adolc.trace_on(adolcID)
    ax = adolc.adouble(init.flatten())
    adolc.independent(ax)
    af = action(ax, beta)
    adolc.dependent(af)
    adolc.trace_off()

    print "taped =", time.time()-start,"s"


    #Test Gradient numerically 
    #    flat = init.flatten()
    #    grad = np.zeros_like(flat)
    #    cost1 = alglib_func(flat,grad,1)
    #    grad2 = grad.copy()
    #    for i in range(len(flat)):
    #        perturb = flat.copy()
    #        perturb[i] = perturb[i]+0.00001
    #        cost2 =alglib_func(perturb,grad2,1)
    #        numgrad = (cost2-cost1)/0.00001
    #        print numgrad/grad[i]

    m = af.shape[0]
    state = xalglib.minlmcreatevj( m ,list(init.flatten()))
    xalglib.minlmsetcond(state,epsg, epsf,epsx,maxits)
    xalglib.minlmoptimize_vj(state, lm_func, lm_jac)
    final, rep = xalglib.minlbfgsresults(state)
    print "optimized: ", time.time()-start, "s"
    print "Exit flag = ", rep.terminationtype, rep.iterationscount
    print "Action = ", action(final,beta)
    return rep.iterationscount, action(final,beta), final
    
if __name__ == "__main__" :

    N = 150
    D = 5
    dt = 0.01
    NBETA = 30
    measIdx = [0]
    taus = [10,15,20]

    M = len(measIdx)
    ytmp = np.loadtxt("dataN_D{0}_dt{1}_noP.txt".format(D,dt))
    # set unmeasured states to NaN, to avoid accidentally accessing
    # them
    
    y = ytmp[:N,measIdx]

    model = lorenz96
    map = rk4

    
    Ntd = len(taus)
    adolcID = 0
    
    store = np.zeros((NBETA,N*D+3))
    x0 = 'start'
    for beta in range(NBETA):
        store[beta,0] = beta
        store[beta,1], store[beta,2], store[beta,3:] = run(N,D,dt,beta,x0)
        x0 = store[beta,3:]
    
    np.savetxt('test_annealRK4.txt', store)
                               
    
    
    
