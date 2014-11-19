import numpy as np
import xalglib, adolc
import time, sys
from scipy.integrate import odeint

# Problem definition
N = 300
D = 5
dt = 0.01
NBETA = 30
measIdx = [0]
taus = [20]

modelname = 'lorenz96'
mapname = 'rk2'
outfile = 'test_BFGS.txt'
generate_data = False
init_path_file = 'initpaths.txt' #if random will generate initpaths.txt, else loads file

epsg = 1e-8
epsf = 1e-8
epsx = 1e-8
maxits = 10000

def lorenz96(x, t):
    D = len(x)
    dxdt = []
    for i in range(D):
        dxdt.append(x[np.mod(i-1,D)]*(x[np.mod(i+1,D)]-x[np.mod(i-2,D)]) - x[i] + 8.17)
    
    return np.array(dxdt)

# Just a convenience to generate data. 
def gen_data(dt, skip=0, data_initial=0, model=lorenz96):

    T_final = (skip+110000)*dt
    T_total = np.arange(0,T_final,dt)
    #data_initial = [ -2.33211 ,  -5.508487 , -9.208057, -10.98852 ,   6.165695]
    if data_initial ==0:
        data_initial = 10.0*np.random.rand(D)

    #[0.80, 0.95, 0.71, 0.24, 0.63];
    Y = odeint(model,data_initial,T_total)
    Y = Y[skip:skip+100001,:]
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

#maps each row in an array x forward to an array f(x)
def array_map(x):
    f = np.zeros_like(x)
    for i in range(x.shape[0]):
        f[i,:] = map(x[i,:],model,dt,i*dt)
    return f

# Evaluates cost function, per variable
def action(x, beta):
#    x = np.array(x)
    x = np.reshape(x,(N,D))
    Rf = 0.01*(2**beta)
    Rm = 4.0
    Rtd = 1.0/(1.0/Rf+1.0/Rm)

    if x.shape != (N,D):
        raise "x is wrong dims!"

    # Rm term
    dy = x[:,measIdx]-y
    action = np.sum(0.5*Rm*(dy)**2)
    
    #Rf term
    f1 = array_map(x)
    action += np.sum(0.5*Rf*(x[1:,:]-f1[:-1,:])**2)
    
    #Rtd term
#    delayMap = np.zeros((len(taus), N, D))

    for count in range(Ntd):
        tau = taus[count]
        if count==0:
            for i in range(1,tau):
                f1 = array_map(f1)
        else:
            for i in range(taus[count-1],tau):
                f1 = array_map(f1)
        dytd = y[tau:,:]-f1[:-tau,measIdx]   
        #        dxtd = x[tau:,measIdx]-f1[:-tau,measIdx]   
        #   + np.sum(0.5*Rf*(dxtd)**2)
        action += np.sum(0.5*Rtd*(dytd)**2) 

    return action/nvar
   

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

# ALGLIB bfgs cost func. returns f(x), passes grad_f(x) by reference
def alglib_func(x,grad,p):
    grad[:] = adolc.gradient(adolcID,x)    
    return  adolc.function(adolcID,x)


def optimize(N,D,dt,beta,x0):
    
    if x0=='start':
        if init_path_file == 'random':
            init = 20.0*np.random.random_sample((N,D))-10.0
            np.savetxt('initpaths.txt',init)        
        else:
            init = np.loadtxt(init_path_file)       
    else:
        init = x0.reshape(N,D)

    #init = np.loadtxt("data_D{0}_dt{1}_noP.txt".format(D,dt))[:N,:]
    
    if init.shape != (N,D):
        raise "x is wrong dims!"  

    ### Use ADOL-C to generate trace, used for derivatives and
    ### evaluations
    start = time.time()

    adolc.trace_on(adolcID)
    ax = adolc.adouble(init.flatten())
    adolc.independent(ax)
    af = action(ax, beta)
    adolc.dependent(af)
    adolc.trace_off()

    
    print "beta = ", beta, " taped"

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

    state = xalglib.minlbfgscreate(1,list(init.flatten()))
    xalglib.minlbfgssetcond(state,epsg, epsf,epsx,maxits)
    xalglib.minlbfgsoptimize_g(state, alglib_func)
    final, rep = xalglib.minlbfgsresults(state)
    print "optimized: ", time.time()-start, "s"
    print "Exit flag = ", rep.terminationtype, rep.iterationscount
    print "Action = ", action(final,beta)
    return rep.terminationtype, action(final,beta), final

def init():
    Ntd = len(taus)
    nvar = D*N
    model = eval(modelname)
    map = eval(mapname)

    if generate_data:
        gen_data(dt,1000,data_initial=0,model=model)

    ytmp = np.loadtxt("dataN_D{0}_dt{1}_noP.txt".format(D,dt))
    y = ytmp[:N,measIdx]

    if len(sys.argv)>1:
        adolcID = sys.argv[1]
    else:
        adolcID = 0 

    store = np.zeros((NBETA,N*D+3))
    x0 = 'start'

def run():
    for beta in range(NBETA):
        store[beta,0] = beta
        store[beta,1], store[beta,2], store[beta,3:] = optimize(N,D,dt,beta,x0)
        x0 = store[beta,3:]
    
    np.savetxt(outfile, store)
    
    
if __name__ == "__main__" :    
    init()
    run()
    
    
    
