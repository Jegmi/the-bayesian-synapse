"""
@author: Jannes Jegminat

Script for running the Bayesian Synapse and plot results.

Important classes and objects:
    Variables : contains all variables
    gen_tab : table where rows contain parameters and simulation results
    plt_figs : takes simulation table and figures
    __main__ : specification of simulation hyperparameter in dictionary "mp",
               run simulation loop and save results in table, call plt_figs
                                  
"""

# =============================================================================
#  plotting? approx linear <var delta> better?
# =============================================================================


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import itertools as it
import pandas as pd
from time import time
from scipy.optimize import fmin
matplotlib.rcParams.update({'font.size': 15})
fontsize = 15
import os
import pickle
import argparse
from collections import OrderedDict

# plt.rcParams["font.family"]

# =============================================================================
#         Helper functions
# =============================================================================    

eta = np.random.randn

def get_axis_and_fig_size():
    ax = plt.gca()        
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    print('axis (w,h):',[width,height])
    print('fig size:',fig.get_size_inches()*fig.dpi)

def panel_label(label, x, ax,y=1.0):
    ax.text(x, y, f'\\textbf{{{label}}}', fontsize=23, 
            transform=ax.transAxes, fontweight='extra bold')

# rc = {
#     "font.family": 'sans-serif',
#     "text.usetex": True,
#     "text.latex.preamble": r'\usepackage{sfmath}',  # allows for sans-serif numbers
#     "axes.spines.right": False,
#     "axes.spines.top": False,
#     "axes.labelsize": 20,
#     "axes.titlesize": 20,
#     "xtick.labelsize": 20,
#     "ytick.labelsize": 20,
#     "lines.linewidth": 2,
#     "legend.frameon": False,
#     "legend.labelspacing": 0,  # vertical spacing bw legend entries
#     "legend.handletextpad": 0.5,  # horizontal spacing bw legend markers and text
#     "legend.fontsize": 20,
#     "mathtext.fontset": 'cm',  # latex-like math font
#     "savefig.dpi": 500,
#     "savefig.transparent": True
# }



#def set_axis_and_fig_size():    
    #axis: [334.8, 217.44]
    #fig size: [432. 288.]
def set_axis_size(w=4.65*1,h=3.02*1, ax=None):
    """ w, h: width, height in inches """

    # plt.gca().spines.set_position('axes',-0.1)
    # if not ax: ax=plt.gca()
    # l = ax.figure.subplotpars.left
    # r = ax.figure.subplotpars.right
    # t = ax.figure.subplotpars.top
    # b = ax.figure.subplotpars.bottom
    # figw = float(w)/(r-l)
    # figh = float(h)/(t-b)
    # ax.figure.set_size_inches(figw, figh)
    

def get_nudt(dim):
    """return sorted randomly drawn firing rates """
    nudt = (np.sort(np.exp((0.5*np.log(10))*eta(dim))))[::-1]*dt
    return(nudt)  

def sigm(a):
    """ logistic sigmoidal """
    return(np.log(1 + np.exp(a)))

def Phi(x):
    """ cumulative gaussian fct """
    return(np.erf(x*0.7071067811865475)*0.5 + 0.5)
    
def NoverPhi(x):
    """ derivative of log cumulative gaussian fct: phi'(x) = N(x)/phi(x) """
    z=np.abs(x*0.70710678118654746)
    t=1.0/(1.0+0.5*z)
    ans=t*np.exp(-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
            t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
            t*(-0.82215223+t*0.17087277)))))))))
    out = ans                
    
    # if x > 0: then adjust else keep
    if x > 0:
        out = 2.0*np.exp(z*z) - ans    
    #ii = x > 0.0        
    #out[ii] = 2.0*np.exp(z[ii]*z[ii]) - ans[ii]
    return(0.79788456080286541/out)
    
def save_obj(obj, name, path='./'):    
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path='./'):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# function for loading    
def get_df(out_path,contains_str=None):
    """ load saved plotting 
        args: 
            out_path : dir to search for pkl files
            contains_str : restrict search to files that contain this str
    """
    if contains_str is None: # turn into trivially true condition
        contains_str = '.pkl'
    out = pd.concat([load_obj(file[:-4],out_path) for file in os.listdir(
                out_path) if (file.endswith(".pkl") and contains_str in file)])
    out = out[np.isnan(out.MSE)==False] # drop non runs
    out = out.sort_index()
    out = out.drop_duplicates()
    return(out)

def cost(theta,xplt,yplt):
    return(np.sum((xplt*theta[1] + theta[0] - yplt)**2))
    
def cost3(theta,xplt,yplt):
    return(np.sum( (yplt - theta[1]*(np.sqrt(xplt/sigm(theta[0]) + 1)-1)/xplt)**2 ))
                            
def expspace(eps0,epsf,N=20):
    """ return log-spaced array with N=20 entries in interval [eps0,epsf] """
    return(np.sort(eps0*np.exp( np.linspace(0,1,N)*np.log(epsf/eps0))))    
    

# =============================================================================
#         Variable class with update rules
# =============================================================================    


class Variables(dict):
    """ Dict class containing the variables of the Bayesian Synapse
        Example: select input spikes for first t time steps, dim = 2, 3 and 4
            self['x'][0:t,2:5]
        p['online'] : if true saves memory by storing only two time steps!            
        
        Methods:
            init : (as opposed to __init__) will initialise variables                        
            res : return MSE performance and time of w_tar within 1-sigma
            plt, plt2 : quick display of variable time series
            
        Update methods (bayesian & classical, approximated & exact)
            linear : updates according to the "linear" feedback model
            binary : update according to the "cerebellar" feedback model
            rl : updates according to the "reinforcement learning" feedback                        
            
    """

    def __init__(self,*arg,**kw):
        # inherit from dict-class
        super(Variables, self).__init__(*arg, **kw)    
        
    def plt(self,key,dim=0,downSample=1):
        d=downSample
        if key in ('wm','lm'):
            m,s2 = ('m','s2') if key=='wm' else ('mu','sig2')
            tspan = (np.arange(0,t_num)*dt)[::d]
            plt.plot(tspan,self[m][::d,dim],'b',label='theo'+str(dim))
            plt.plot(tspan,self[m][::d,dim]+self[s2][::d,dim]**0.5,':b')
            plt.plot(tspan,self[m][::d,dim]-self[s2][::d,dim]**0.5,':b')
            plt.plot(tspan,np.exp(self['lam'][::d,dim]),'r') if key=='wm' else plt.plot(tspan,self['lam'][::d,dim],'r')
        else:
            if len(self[key].shape) == 1:
                plt.plot(np.arange(0,t_num)*dt,self[key],label=key)
            else:
                plt.plot(np.arange(0,t_num)*dt,self[key][:,dim],label=key+str(dim))
            plt.gca().legend()
            plt.xlabel('time')
            plt.ylabel(key)

    def plt2(self,key,dim=0,downSample=1,lw=2,k_cut=0,q=2):
        """ fancy version of plt """
        d=downSample
        if key in ('wm','lm'):
            m,s2 = ('m','s2') if key=='wm' else ('mu','sig2')
            tspan = (np.arange(0,t_num-1-k_cut)*dt)[::d]
            # ground truth
            yplt = np.exp(self['lam'][k_cut:-1:d,dim]) if key=='wm' else self['lam'][::d,dim]
            plt.plot(tspan,yplt,'k',linewidth=lw)
            # filter
            yplt, err = self[m][k_cut:-1:d,dim], self[s2][k_cut:-1:d,dim]**0.5            
            plt.fill_between(tspan,yplt-q*err,yplt+q*err,alpha=0.3,color='r')
            plt.plot(tspan,yplt,'r',linewidth=lw)                                    
        else:
            if len(self[key].shape) == 1:
                plt.plot(np.arange(0,t_num)*dt,self[key],label=key)
            else:
                plt.plot(np.arange(0,t_num)*dt,self[key][:,dim],label=key+str(dim))
            plt.gca().legend()
            plt.xlabel('time')
            plt.ylabel(key)

    def init(self,t_num,p,mp):
        """ init variables, add new ones if needed """
        dim = p['dim']
        v = self
        # constants
        v['m_ou'] = np.exp(p['mu_ou'] + p['sig2_ou']/2)
        v['s2_ou'] = v['m_ou']**2*(np.exp(p['sig2_ou'])-1)
        # approx sig2_u as constant         
        s2_var = 2*v['s2_ou'] if np.isnan(p['k']) else v['s2_ou']+v['m_ou']*p['k']
        v['<sig2u>'] = p['nu*dt'].dot(1-p['nu*dt'])*s2_var + p['sig0u']**2        
        v['lam'] = np.zeros((t_num,dim)) # ground truth
        v['lam'][0] = np.ones(dim)*p['mu_ou']
        v['mu'] = np.zeros((t_num,dim))
        v['sig2'] = np.zeros((t_num,dim))
        v['mu'][0] = np.ones(dim)*p['mu_ou']*p['init_deviation']
        v['sig2'][0] = np.ones(dim)*p['sig2_ou'] if np.isnan(p['lr']) else np.ones(dim)*p['lr']
        # weight space
        v['w'] = np.zeros((t_num,dim)) # sampled
        v['m'] = np.ones((t_num,dim)) 
        v['m'][0] = v['m_ou']
        v['s2'] = np.ones((t_num,dim))
        v['s2'][0] = v['s2_ou']
        v['x'] = np.zeros((t_num,dim))
        v['u'] = np.zeros(t_num)  
        v['sig2u'] = np.zeros(t_num)
        v['sig2u'][0] = v['<sig2u>']

    def rl(self,p,k,t):
        """ update for RL based on parameters p, pointer k and time t """
     
        # feedback signal
        f = - np.abs(v['u'][k])
        
        if p['bayesian']:            
            # compute synapse variance vector
            s2_var = 2*v['s2'][k] if np.isnan(p['k']) else v['s2'][k]+v['m'][k]*p['k']
            
            if p['approx_method']==0:
                v['sig2u'][k] = s2_var.dot(v['x'][k]**2) + p['sig0u']**2
                term1 = v['sig2'][k]*v['m'][k]*v['x'][k]/(v['<sig2u>'] if p['const_pref'] else v['sig2u'][k])
                z =  f*v['x'][k]*(v['m'][k] - v['w'][k])/v['sig2u'][k]
                dmu_like = term1*(f*np.tanh(z) - v['x'][k]*(v['m'][k] - v['w'][k]))
                dsig2_like = term1**2*((f/np.cosh(z))**2 -v['sig2u'][k])

            # linearisation of tanh and cosh
            if p['approx_method']==1:
                v['sig2u'][k] = s2_var.dot(v['x'][k]**2) + p['sig0u']**2
                term1 = v['sig2'][k]*v['m'][k]*v['x'][k]/(v['<sig2u>'] if p['const_pref'] else v['sig2u'][k])
                glob = f**2/v['sig2u'][k] - 1
                dmu_like = term1*(v['m'][k] - v['w'][k])*glob
                dsig2_like = term1**2*glob*v['sig2u'][k]            
            
            return(dmu_like,dsig2_like)

        # classical
        else:
            #sig2_u = p['sig0u']**2 # 1/sig2_u absorbed in learning rate
            term1 = v['sig2'][k]*v['x'][k]#/sig2_u
            z =  f*v['x'][k]*(v['m'][k] - v['w'][k]) 
            # rm v['x'][k] cause x^2 = x
            dmu_like = term1*(f*np.tanh(z) - (v['m'][k] - v['w'][k])) 
            dsig2_like = 0            
            return(dmu_like,0)
                            
    def binary(self,p,k,t):
        """ update for binary rule, similar structure to rl-method
        """

        if p['bayesian']:            
            s2_var = 2*v['s2'][k] if np.isnan(p['k']) else v['s2'][k]+v['m'][k]*p['k']            
            if p['approx_method']==0 or t < p['k_cut']/2:
                v['sig2u'][k] = s2_var.dot(v['x'][k]**2) + p['sig0u']**2
            # Constant
            if p['approx_method']==1 and t >= p['k_cut']/2:
                v['sig2u'][k] = v['<sig2u>']                

            term1 = v['sig2'][k]*v['m'][k]*v['x'][k]/np.sqrt(v['sig2u'][k])
            f = np.sign(v['u'][k] - p['u0']*p['sig0u'])
            z = -f*p['u0']*p['sig0u']/np.sqrt(v['sig2u'][k])
            term2 = NoverPhi(z)
            dmu_like = term1*f*term2
            dsig2_like = - term1**2*term2*(term2 + z)
            
            return(dmu_like,dsig2_like)
        # classical
        else:
            sig2u = np.sqrt(p['sig0u']**2)
            term1 = v['sig2'][k]*v['x'][k] #/sig2u absorbed in learning rate
            f = np.sign(v['u'][k] - p['sig0u']*p['u0'])
            z = -f*p['u0']*p['sig0u']/sig2u
            term2 = NoverPhi(z)
            dmu_like = term1*f*term2                        
            return(dmu_like,0)
            
    def linear(self,p,k,t):
        """ update for linear rule, similar structure to rl-method
        """
        if p['bayesian']:                        
            s2_var = 2*v['s2'][k] if np.isnan(p['k']) else v['s2'][k]+v['m'][k]*p['k']
            
            if p['approx_method']==0 or t < p['k_cut']/2:
                v['sig2u'][k] = s2_var.dot(v['x'][k]**2) + p['sig0u']**2
            # Constant
            if p['approx_method']==1 and t >= p['k_cut']/2:
                v['sig2u'][k] = v['<sig2u>']

            term1 = v['sig2'][k]*v['m'][k]*v['x'][k]/v['sig2u'][k]
            dmu_like = term1*v['u'][k]
            dsig2_like = -term1**2*v['sig2u'][k]
            
            return(dmu_like,dsig2_like)

        # classical
        else:            
            #1/p['sig0u']**2 absorbed in leraning rate
            term1 = v['sig2'][k]*v['x'][k] 
            dmu_like = term1*v['u'][k]
            return(dmu_like,0)                    
            
        
    def res(self,method='lam',k=0,fac=2):
        """ compute performance measures: MSE and fraction of being inside 
            k = cutoff
        """
        if 'lam' in method:
            delta = (self['mu'][k:] - self['lam'][k:])**2
            return(np.mean(delta),np.mean(delta < fac**2*self['sig2'][k:]))
        elif 'w' in method:
            delta = np.mean((self['m'][k:] - self['w'][k:])**2)
            return(np.mean(delta),np.mean(delta < fac**2*self['s2'][k:]))

# =============================================================================
#         Simulation table
# =============================================================================    

def gen_tab(mp):
    """ initialise DataFrame with simulation meta parameters 
        and keep placeholders for simulation results
    
        args: 
            mp : dict containing parameters, e.g., mp['lr'] is a list of learn-
                 ing rates or mp['rules'] for all learning rules simulated
            
        output:
            DataFrame : nan-values to be filled by the simulation
    """
    
    
    ### loop over all parameters and write into DataFrame
    out = []
    # if parameter dictionary is specified for Bayesian rule but not for the 
    # classical rule, add an additional entry with "grad-dict" and same vals
    for dic in [key for key in mp.keys() if '-dict' in key]:
        if set(mp['rules']).issubset(set(mp[dic].keys())) == False: # elements missing        
            for rule in [rule for rule in mp['rules'] if rule not in mp[dic]]:
                mp[dic][rule] = mp[dic][rule[:-5]]
    # loop parameters
    A,B,C,D,E,F,G,H,I,J,K,M,N,O,T,Q,R = ('lr','w-dynamic','k','sample_method',
                             'approx_method','MSE','p_in','rule',
                             'mu_ou','mu_ou-gm','sig2_ou','sig2_ou-gm',
                             'sig0u','sig0u-gm','tau_ou','const_pref','dim') # 14
    # run Bayesian sim across whatever that means
    for pset in it.product(mp['w-dynamics'],mp['ks'],mp['sample_methods'],
                           mp['approx_methods'],[np.nan],[np.nan],mp['rules'],
                           mp['mu_ous'],mp['mu_ous-gm'],mp['sig2_ous'],
                           mp['sig2_ous-gm'],mp['sig0us'], mp['sig0us-gm'],
                           mp['const_prefs']):
        b,c,d,e,f,g,h,i,j,k,m,n,o,q = pset
        t = mp['tau_ou-dict'][h] # tau_ou
        r = mp['dim-dict'][h]
        for a in mp['lrs-dict'][h]:
            out.append({A:a,B:b,C:c,D:d,E:e,F:f,G:g,H:h,I:i,
                        J:j,K:k,M:m,N:n,O:o,T:t,Q:q,R:r})
    out = pd.DataFrame(out)
    
    ### clean up by removing redundant options, e.g., Bayesian + learning rate    
    # rm learning rates for Bayesian rules
    out.loc[out.rule.apply(lambda r: ('-Grad' not in r)),'lr'] = np.nan
    # remove approximation method for grad
    out.loc[out.rule.apply(lambda r: ('-Grad' in r)),'approx_method'] = np.nan
    # if grad + posterior sampling => set to no sampling
    out.loc[(out.rule.apply(lambda r: ('-Grad' in r)) & np.isnan(out.k)),'sampling'] = np.nan
            
    # rm const prefactor for all rules (including RL-grad), only keep for RL
    out.loc[out.rule!='RL','const_pref'] = np.nan
    
    # rm sample method np.nan for RL, add 'gauss' instead
    out.loc[out.rule.apply(lambda r: ('RL' in r)),'sample_method'] = 'gauss'
    
    # match -gm if not specified explicelty
    for kk in [kk for kk in out.keys() if '-gm' in kk]: #
        out.loc[np.isnan(out[kk]),kk] = out.loc[np.isnan(out[kk]),kk[:-3]]                        
    out = out.drop_duplicates()
    out.reset_index(inplace=True)
    out.drop(['index'], axis=1,inplace=True)    
    return(out)
    
def add_key(tab,key,values):
    """ augment a DataFrame by "tensorial multiplication"
        
        args:
            tab : DataFrame of length L
            key : name of new column, i.e., name of the new index in a tensor
            values : array or list, i.e., values of the new index
        
        output:
            out : DataFrame of length L*len(values)    
    """
    out = []
    for val in values:
        for i in range(len(tab)):
            mydict = dict(tab.iloc[i].items())
            mydict.update({key:val})
            out.append(mydict)
    out = pd.DataFrame(out)
    return(out)    
        
# =============================================================================
#         Plotting, including helpers
# =============================================================================    
    
    
def plt_legend(loc=None,ncol=1,prop={}):
    """ legend without double entries """
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc=loc,ncol=ncol,prop=prop)


class Fit(object):
    """ regression on a dataset of inputs X and labels Y 
        with arbitrary function fct 
        quadratic cost and basic plotting utility. 
        log=True: independent variable X cast into log space for fit & plot """
    def __init__(self,x,y,log=False):
        """ load data X, Y arrays """
        self.x=x
        self.y=y
        self.log = log
        self.T = lambda x:x if log==False else np.log
    def fit(self,fct,th0):  
        """ define fitting fct, init pars and use quadratic cost """
        self.fct = fct
        self.th = fmin( lambda th: np.sum((fct(self.T(self.x),th) 
                                            - self.T(self.y))**2), th0)
    def plt(self,c='k',lw=None,label='Theory',alpha=1,marker=None):        
        """ """
        self.T_inv = lambda x: x if self.log==False else np.exp
        xlim = plt.gca().get_xlim()
        xlin = np.linspace(xlim[0],xlim[1])
        ylin = self.T_inv(self.fct(xlin,self.th))
        plt.plot(xlin,ylin,c=c,label=label,lw=lw,alpha=alpha,marker=None)

        
def plt_set_logxticks(xticks,ax=None):    
    """ use log x axis and place ticks at exact locations """
    ax = plt.gca() if ax is None else ax
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(xticks)
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.get_xaxis().set_major_formatter(formatter)

def plt_set_logyticks(yticks,ax=None):    
    """ use log x axis and place ticks at exact locations """
    ax = plt.gca() if ax is None else ax
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(yticks)
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.get_yaxis().set_major_formatter(formatter)

def rotate_xticks(rotation=0):
    """ rotate xticks in bar plot (to ensure readibility) """
    for ax in plt.gcf().axes:
        plt.sca(ax)
        plt.xticks(rotation=rotation)
# legend
def boxplot_legend(labels,colors,alpha=0.6,loc=None,ncol=1):
    """ draw custom legend for box plots with labels & colors same length! """
    handels = [matplotlib.patches.Patch(color=c,label=m,alpha=alpha
                                        ) for c,m in zip(colors,labels)]
    plt.legend(handles=handels,loc=loc,ncol=ncol)#, bbox_to_anchor=(0.5, 0.5)) #, 
           #loc='center', ncol=2, facecolor="white", numpoints=1 )

# fancy plotter for time series
def plt_timeseries(v,key,dim=0,lw=2,q=2):
    """ fancy version of plt for timeseries plotting """
    if key in ('wm','lm'):
        m,s2 = ('m','s2') if key=='wm' else ('mu','sig2')
        # ground truth
        yplt = np.exp(v['lam'][:,dim]) if key=='wm' else v['lam'][:,dim]
        plt.plot(v['tspan'],yplt,'k',linewidth=lw)
        # filter
        yplt, err = v[m][:,dim], v[s2][:,dim]**0.5            
        plt.fill_between(v['tspan'],yplt-q*err,yplt+q*err,alpha=0.3,color='r')
        plt.plot(v['tspan'],yplt,'r',linewidth=lw)

      
def plt_figs(fig,mp,lw=3,fontsize=18,plt_path='./',res_path='./'):
    """ main plotting fct: generate fig in manuscript and save in plt_path.
        data for figs is loaded from res_path. 
        
        Change appearance of plots below!
        
        fig 2: generate time series plots for learning rules and sim_ids (many)
        fig 3: generate three MSE plots (one for each learning rule)
        fig 4: generate three plots: firing rate vs learning rate vs variance
        fig 5X: generate three MSE robustness plots for prior hyperpars
        fig 6: generate single histogram plot, MSE robustness wrt to prior dyn.
        
    """
    
    matplotlib.rcParams.update({'font.size': fontsize})
    mp['r2t'] = {'RL':'RL','Linear':'Linear','Binary':'Cerebellar'}     
    
    if fig == 2:
        f,axs = plt.subplots(2,3,figsize=(10,4),constrained_layout=False,
                                                             sharey=False)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.9)
        
        for r,ax,axlabs in zip(['Linear','Binary','RL'],
                              axs.T,(('a','d'),('b','e'),('c','f'))):
            
            # load
            v_ts = load_obj('fig_2_rule_{0}'.format(r),path=res_path)
            tau_ou = mp['tau_ou-dict'][r]            
            plt_dim=mp['plt-dim'][mp['plt-dim']<(100 if r=='RL' else 1000)]             
            #for i_dim in range(len(plt_dim)):
            for i_dim,ax_i,i in zip([0,len(plt_dim)-1],ax,(0,1)): # first and last
                plt.sca(ax_i)
                [plt.locator_params(axis=axis, nbins=2) for axis in ['x','y']]                                    
                plt_timeseries(v_ts,'wm',dim=i_dim,lw=lw,q=2)                
                #plt.xlabel('time s')                
                plt.ylim([0,1.5])
                #tau_ou = 1000 if r != 'RL' else 5000
                plt.gca().set_xticks([0,tau_ou,2*tau_ou,3*tau_ou])
                plt.xlim([0,3*tau_ou])
                plt.title(mp['r2t'][r] + r', $\nu = $' + str(round(
                                v_ts['nu*dt'][i_dim]/dt,2)) + ' Hz')

                # suppress y-ticks & label for RL and binary:
                if 'Linear' in r:
                    if i==0:
                        plt.ylabel('weight mV')
                        plt.gca().axes.yaxis.set_ticklabels([0,1])
                        plt.gca().axes.xaxis.set_ticklabels([])
                    else:
                        plt.xlabel('time s')
                        plt.gca().axes.yaxis.set_ticklabels([])
                elif r in ('RL','Binary') and i==1:                    
                    plt.gca().axes.yaxis.set_ticklabels([])
                else:
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.gca().axes.yaxis.set_ticklabels([])                    
                panel_label(axlabs[i],-0.22 if r=='Linear' else -0.15,ax_i,y=1.4)
                    

        plt.savefig(plt_path + 'fig_{2}_rule_{1}_TS_{0}.pdf'.format(
                        plt_dim[i_dim],r,fig),dpi=300,bbox_inches='tight')
        plt.show(),plt.close()

    # fig 3: MSE, not on cluster
    if fig == 3 and my_args.i == -1: # MSE, not on cluster
        out = get_df(res_path,contains_str='fig_3')
        y = 'MSE'        
        ymax = {'RL':0.25, 'Linear':0.125, 'Binary':0.25}
        f,axs = plt.subplots(1,3,figsize=(9,4),constrained_layout=True)
        rules = np.array([r for r in out.rule.unique() if '-Grad' not in r])
        for r,i in zip(rules[[1,2,0]],range(3)):
            out2 = out[out.rule.apply(lambda ri: (r in ri))].sort_values('lr')

            plt.sca(axs[i])
            
            bMSE = out2[out2.rule==r][y].values[0]
            bxplt = out[out.rule=='Linear-Grad']['lr'].values
            cMSE = out2[out2.rule==r+'-Grad'][y].values
            xplt = out2[out2.rule==r+'-Grad']['lr'].values

            plt.plot(xplt,cMSE,'k',marker='o',lw=lw,label='Classic')            
            plt.plot(bxplt,bMSE*np.ones(len(bxplt)),'r',ls='--',lw=lw,
                     label='Bayesian')
            
            if mp['log_scale']:
                plt_set_logxticks([0.0001,0.001,0.01])
                plt.xlim([0.0001,0.1])
            #plt.ylim([0,ymax[r]])
            [plt.locator_params(axis=axis, nbins=2) for axis in ['y']]            
            plt.xlabel(r'learning rate $\eta$')
            plt.gca().set_yticks([0,0.1])
            plt.ylim([0,0.16])
            
            # suppress for RL and binary:
            if 'Linear' in r:
                get_axis_and_fig_size()
                plt.gca().legend(loc=2,ncol=1,prop={'size':18})#loc=1 if r == 'RL' else 4)                        
                plt.ylabel(y)
            else:
                plt.xlabel('')
                plt.gca().axes.xaxis.set_ticklabels([])
                plt.gca().axes.yaxis.set_ticklabels([])
            #     set_axis_size()
                                 
            panel_label(('a','b','c')[i], (-.22,-0.15,-0.15)[i], axs[i],y=1.15)
                
            plt.title(mp['r2t'][r])
        plt.savefig(plt_path + 'fig3_MSE_rule_{0}.pdf'.format(r),
                    dpi=300,bbox_inches='tight')
        plt.show(), plt.close()

                
    # fig 5 # robustness (keep gm const), not on cluster
    elif '5' == str(fig)[0] and my_args.i == -1:   
        
        id2lab = {1:r'prior mean, $m_{\rm{prior}}/m_{\rm{prior, theory}}$', 
                  2:r'prior variance, $s^2_{\rm{prior}}/s_{\rm{prior, theory}}^{2}$',
                  3:r'membrane noise, $\sigma_{0}/\sigma_{0, \rm{theory}}$',}        
        
        id2tit = {1:r'$m_{\rm{prior, theory}} = -0.669$', 
          2:r'$s_{\rm{prior, theory}}^{2} = 0.07448$',
          3:r'$\sigma_{0, \rm{theory}} = 2$'}        
        
        r2c = {'Linear':'r','Binary':'gray','RL':'k'}
        #r2loc = {'Linear':'r','Binary':'gray','RL':'k'}

        for plt_id in [1,2,3]: # hyperparameters

            out = get_df(res_path,contains_str='fig_5{0}'.format(plt_id))
            x = {1:'mu_ou', 2:'sig2_ou', 3:'sig0u'}[plt_id]            
            y = 'MSE'                    
                        
            for r in out.rule.unique():
                out2 = out[out.rule==r].sort_values(x)
                x0 = out2[x + '-gm'].unique()[0]
                xplt = out2[x].values/x0
                bMSE = out2[y].values
                plt.plot(xplt,bMSE,c=r2c[r],marker='o',lw=lw,label=mp['r2t'][r])
                                        
            # y-scale and vline
            if plt_id == 3:
                plt.ylim([0,0.2])
            else:
                plt.ylim([0,plt.gca().get_ylim()[1]*1.1])
            [plt.locator_params(axis=axis, nbins=2) for axis in ['y']]
            ylim = plt.gca().get_ylim()
            plt.plot([1,1],ylim,c='black',alpha=0.6,ls='--',lw=lw*0.75,label='Theory')# {0}'.format(r),lw=2)                                
            
            ncol = 1 #2 if plt_id == 1 else 1
            plt.gca().legend(ncol=ncol,loc= None if plt_id < 3 else 2,
                   prop={'size':16})
            plt.xlabel(id2lab[plt_id])            
            plt.ylabel(y)
            plt.title(id2tit[plt_id] + str())
            plt_set_logxticks([0.5,1,2])
            
            plt.savefig(plt_path + 'figS2_MSE_robustness_{0}.pdf'.format(x),
                        dpi=300,bbox_inches='tight'),plt.show(),plt.close()
            plt.show(), plt.close()

    elif fig == 6 and my_args.i == -1: ## robustness vs OU
        # load
        out = get_df(res_path,contains_str='fig_6')
        y = 'MSE'
        out2 = out.iloc[[0,3,1,4,2,5]]
        out2.loc[out2.rule=='Binary','rule'] = 'Cerebellar'
        out2[y].plot.bar(color=['r','r','gray','gray','k','k'],alpha=0.6)
        
        hatches = [' ','x',' ','x',' ','x']
        [bar.set_hatch(hatch) for bar, hatch in zip(plt.gca().patches,hatches)]
        plt.ylabel(y)
        [plt.locator_params(axis=axis, nbins=2) for axis in ['y']]        
        plt.gca().set_xticklabels(out2['w-dynamic'])
        rotate_xticks()         
        boxplot_legend(out2.rule.unique(),['r','gray','k'],alpha=0.6,loc=4)
        plt.title(r'Robustness: prior dynamics')        
        plt.savefig(plt_path + 'fig6_MSE_robustness_OU.pdf',dpi=300,
                                                        bbox_inches='tight')
        plt.show(),plt.close()
        

    # fig 4: supp plots and fits
    elif fig == 4 and my_args.i == -1:

        s = 15
        mp['r2t'] = {'RL':'RL','Linear':'Linear','Binary':'Cerebellar'}
        
        lab = {}
        lab['s2_m'] = r'norm. variability, $\langle \sigma^2_i/ \mu_i \rangle_t$'
        lab['dm_m'] = r'norm. learn. rate, $\langle |\Delta \mu_i|/\mu_i \rangle_t$'
        lab['nu'] = r'input firing rate, $\nu_i$'
                
        # derive label from fit
        labf = lambda a,nu0,n=3 : 'Fit: ' + r'$a=$' + str(round(a,n)
                                        ) + r', $\nu_0=$' + str(round(nu0,n))
        
        # load dict in alphabetical order
        outs = [load_obj(file[:-4],res_path) for file in 
             sorted(os.listdir(res_path)) if (file.endswith(".pkl") and 'fig_4' in file)] # and file.startswith("Supp")]
        # run rules in alphabetical order
        rules = ['Binary','Linear','RL'] # [r for r in mp['rules'] if '-Grad' not in r]
        for r,out in zip(rules,outs):            
            
            dm_m = out['dm_m']
            s2_m = out['s2_m']            
            nu = out['nu']
                        
            ### Fig 1: x=s2/m y=dm/m ###
    
            ## plot
            T = np.log if mp['log_scale'] else lambda x:x
    
            xplt, yplt = s2_m, dm_m
            plt.scatter(s2_m,dm_m,color='k',alpha=0.5,s=s,label='Simulation')
            plt.xlabel(lab['s2_m'])
            plt.ylabel(lab['dm_m'])
            plt.title('{0}'.format(mp['r2t'][r]))

            slope = 1               
            out = fmin(lambda th: cost([th,slope],T(xplt),T(yplt)),[0])
            if mp['log_scale']:
                plt.plot(xplt,xplt**slope*np.exp(out[0]),lw=lw,c='r',label='Theory: slope 1')
            else:
                plt.plot(xplt,xplt*slope + out[0], c='r',label='Theory, slope 1/2')
            
            ## make nice and save
            plt.gca().legend()
            if mp['log_scale']:
                plt_set_logxticks([0.005,0.01,0.05])
                plt_set_logyticks([0.001,0.01])
                if r == 'RL':
                    plt_set_logyticks([0.001,0.002,0.003])
                    plt.ylim([0.001,0.003])
                else:
                    plt.xlim([0.003,0.05])
            print(plt.gca().get_xlim())
            plt.savefig(plt_path + 'dm_m_vs_s2_m_{0}.pdf'.format(r),dpi=300,bbox_inches='tight')
            plt.show(),plt.close()
        
            ### Fig 2: x=nu y=dm/m ###                
            T = np.log if mp['log_scale'] else lambda x:x
    
            ## plot 
            xplt, yplt = nu/dt, dm_m
            plt.scatter(xplt,yplt,color='k',alpha=0.5,s=s,label='Simulation')                            
            plt.xlabel(lab['nu']) # plt.xlabel(r'Input rate $\nu_i$')
            plt.ylabel(lab['dm_m'])
            plt.title('{0}'.format(mp['r2t'][r]))
            fit = Fit(xplt,yplt)
            fit.fit(lambda x,th: th[0]/x*( (xplt/sigm(th[1]) + 1)**0.5 - 1),[1,1])
            a, nu0 = fit.th[0], sigm(fit.th[1])
            yplt_fit = a/xplt*( (xplt/nu0 + 1)**0.5 - 1)
            plt.plot(xplt,yplt_fit,c='r',label=labf(a,nu0),lw=lw)            
            plt_set_logxticks([0.1,1,10])
            if r=='RL':
                plt_set_logyticks([0.001,0.005])
                plt.ylim([0.001,0.005])
            else:
                plt_set_logyticks([0.001,0.01,0.1])                        
            plt.gca().legend()      
            plt.savefig(plt_path + 'dm_m_vs_nu_{0}.pdf'.format(r),dpi=300,bbox_inches='tight')
            plt.show(),plt.close()
    
            ### Fig 3: x=nu y=s2/m ###
            
            ## plot
            xplt, yplt = nu/dt, s2_m
            plt.scatter(xplt,yplt,color='k',alpha=0.5,s=s,label='Simulation')
            plt.xlabel(lab['nu'])
            plt.ylabel(lab['s2_m'])
            plt.title('{0}'.format(mp['r2t'][r]))
            
            ## add lines (fit only synapses with 2 Hz) TODO: not sure what's needed?
            fit = Fit(xplt,yplt)
            fit.fit(lambda x,th: th[0]/x*( (xplt/sigm(th[1]) + 1)**0.5 - 1),[1,1])
            a, nu0 = fit.th[0], sigm(fit.th[1])
            yplt_fit = a/xplt*( (xplt/nu0 + 1)**0.5 - 1)
            plt.plot(xplt,yplt_fit,c='r',label=labf(a,nu0),lw=lw)            
            plt.gca().set_yscale('log') ,plt.gca().set_xscale('log')
            plt.gca().legend()                        
            
            ## make nice and save
            plt.gca().legend()
            if mp['log_scale']:
                #plt.gca().set_xscale('log')
                plt_set_logxticks([0.1,1,10])
                if r=='RL':
                    plt_set_logyticks([0.02,0.05])
                    plt.ylim([0.02,0.05])
                else:
                    plt_set_logyticks([0.001,0.01,0.1])
            plt.savefig(plt_path + 's2_m_vs_nu_{0}.pdf'.format(r),dpi=300,bbox_inches='tight')
            plt.show(),plt.close()    
    
    
np.random.seed(4) # set random seed
if __name__== "__main__":
    """            
        Overview: parameter defaults are set in dict "mp" and affect all sims.
                  Figure specific options are set below and overwrite defaults.
                  The DataFrame "tab" is generated and contains a sim per row.
                  These sims can be run in parallel on a cluster by an "i"-loop
                  The sim-specific parameters are stored in the dict "p",
                  each sim propagates the variables stored in "v" via updates,
                  the outputs needed to plot "Figure" are saved externally,
                  plt_figs loads these outputs for plotting "Figure".
        
        Args: 
            --Figure : determines which plot is produced:
                       fig = 2: time series
                       fig = 3: MSE plot
                       fig = 4: firing rate vs learning rate vs variance
                       fig = {51, 52, 53} : MSE robustness wrt to parameters
                       fig = 6 : MSE robustness wrt to prior dynamics
            --i : selects single row of simulation table for i>-1; -1 runs all
            --Load : skip sims; load pre-computed results and plot directly.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--Figure","-f",default=3,type=int, 
                        help="Figure to plot")
    parser.add_argument("--i","-i",default=-1,type=int,
                        help="process id on cluster; -1 for local machine")
    parser.add_argument("--Load","-l",default=1,type=int,
                        help="if =1, skip simulation and load previous results")        
        
    my_args = parser.parse_args()
    
    res_path = './pkl_files/' # <--- path for storing and loading sim results
    plt_path = './figs/' # <--- storing plots generated by plt_figs

    # pars
    mp,p = {},{}        
    #### defaults #####    
    mp['rules'] = ['Linear','Binary','RL']
    # bayesian
    mp['sample_methods'] = [np.nan] # 'gauss',np.nan] RL always 'Gauss'
    mp['approx_methods'] = [1] # 0: true value, 1:approx by const, 2: running mean
    mp['const_prefs'] = [False] # only for RL?
    mp['w-dynamics'] = ['OU']
    # if gm = np.nan, use values of non-gm model
    mp['mu_ous-gm'] = [np.nan]
    mp['sig2_ous-gm'] = [np.nan]
    mp['sig0us-gm'] = [np.nan]
    mp['mu_ous'] = [-0.6690]
    mp['sig2_ous'] = [0.07448]
    mp['sig0us'] = [2]
    mp['tau'] = 0.01
    mp['u0']= -2.1 # for binary
    mp['ks'] = [0.0877] #np.nan : posterior sampling
    mp['tau_ou-dict'] = {'RL':5000,'Linear':1000,'Binary':1000}
    mp['dim-dict'] = {'RL':100, 'Linear':1000,'Binary':1000}

    # timing
    dt = 0.01
    mp['cut_steps'] = 2 # OU times cut away
    mp['steps'] = 100 # OU times for sim (including cut steps)
    mp['online'] = True
    
    # for grad rule
    num_lr = 31
    mp['log_scale'] = True
    mp['lrs-dict'] = {'RL-Grad':expspace(0.0001,0.25,num_lr),
                      'Linear-Grad':expspace(0.001,0.1,num_lr),
                      'Binary-Grad':expspace(0.001,0.2,num_lr),
                      'RL': [np.nan], 'Linear':[np.nan], 'Binary':[np.nan]}

    mp['lrs-dict'] = {'RL-Grad':expspace(0.0001,0.1,num_lr),
                      'Linear-Grad':expspace(0.0001,0.1,num_lr),
                      'Binary-Grad':expspace(0.0001,0.1,num_lr),
                      'RL': [np.nan], 'Linear':[np.nan], 'Binary':[np.nan]}
    
    # supp plots
    mp['fit-slope'] = False
    mp['plt-dim'] = np.array([0,1,2,3,10,20,30,99,200,300,500,900])
    mp['down_sample'] = 1
    mp['r2t'] = {'RL':'Reinforcement Leanring','Linear':'Linear','Binary':'Binary'}

    # select figure
    #my_args.Figure = 2
    mp['fig'] = fig = my_args.Figure
    mp['load_previous_results'] = my_args.Load

    # Fig 2: 3 time series plots    
    if fig == 2: # time series
        mp['rules'] = ['Linear','Binary','RL']
        mp['online'] = False # times series won't work in online mode!
        mp['steps'] = 5
        #mp['plt-dim'] = [0,1,2,3,10,20,30,100-1]
        #mp['plt-dim'] = [900]
        mp['down_sample'] = 100
        
    # Fig 3: 3 plots
    elif fig == 3: # MSE (42)
        mp['rules'].extend(['Linear-Grad','Binary-Grad','RL-Grad'])
#        mp['rules'] = ['RL','RL-Grad']
#        print('Rm: only running RL here')
        mp['rules'] = ['Linear','Linear-Grad','Binary','Binary-Grad']
        print('Reminder: only running Linear and Binary here')
        mp['steps'] = 500        
        
    # S1: 3 x 3 plots, including fits
    elif fig == 4: # supps (3)
        mp['ks'] = [np.nan] # posterior sampling
        
    # S2: 3 x 3 plots (33)
    elif fig == 51: # robustness, with model mismatch 
        mp['mu_ous-gm'] = mp['mu_ous']
        mp['mu_ous'] = mp['mu_ous']*expspace(0.5,2,11)
    elif fig == 52: # robustness, with model mismatch           
        mp['sig2_ous-gm'] = mp['sig2_ous']
        mp['sig2_ous'] = mp['sig2_ous']*expspace(0.5,2,11)
    elif fig == 53: # robustness, with model mismatch           
        mp['sig0us-gm'] = mp['sig0us']
        mp['sig0us'] = mp['sig0us']*expspace(0.5,2,11)
                     
    # S3: 3 plots OU robustness# (6)
    elif fig == 6: 
        mp['w-dynamics'] = ['OU','jump']


# =============================================================================
#       Generate simulation table for Figure "fig". Simulate or load. Plot.
# =============================================================================
        
    # iterate through pars in tab and report MSE
    tab = gen_tab(mp)
    
    ## load or simulate
    if mp['load_previous_results']:
        print('Skip sims and load results for Figure:',fig,'in dir:',plt_path)
        sim_ids = []
    else:
        print('Run simulations for Figure =',fig)
        sim_ids = np.arange(len(tab)) if my_args.i==-1 else [
                  np.arange(len(tab))[my_args.i]]        
        
    for sim_id in sim_ids:
        # transfer parameters from mp and tab to p
        p.update(mp)
        p.update(tab.iloc[sim_id])
                        
        # init world parameters        
        dim = p['dim']
        p['k_cut'] = int(p['cut_steps']*p['tau_ou']/dt)
        dW = (dt*p['sig2_ou-gm']/p['tau_ou']*2)**0.5                
        p['nu*dt'] = get_nudt(dim)
        p['plt-dim'] = np.array(p['plt-dim'])[p['plt-dim']<dim] # rm large dims
        while (p['nu*dt']>1).any(): p['nu*dt'] = get_nudt(dim)

        # init variabes 
        p['bayesian'] = np.isnan(p['lr']) # signature of Bayesian
                        
        t_num = int(np.ceil(mp['steps']*p['tau_ou']/dt))
        p['init_deviation'] = (1 + eta(dim))
        v = Variables()
        v.init(2 if mp['online'] else t_num,p,mp)                        
        
        t0,tout,err,supps = time(),1, np.zeros(2),np.zeros([3,dim])
        for t in np.arange(0,t_num-1): # make sure things dont explode
            k = t if mp['online']==False else 0 # k is array index
            # timing
            if time()-t0 > tout*60:
                tout, t_per_min = tout + 1, t if tout==1 else t_per_min
                print(t,'out of',t_num, '; min left for sim_id:',int((t_num-t)/t_per_min))
    
            # update vars
            v['m'][k] = np.exp(v['mu'][k] + v['sig2'][k]/2) if p['bayesian'] else np.exp(v['mu'][k])
            if np.any(v['m'][k] == np.inf) or np.any(np.isnan(v['m'][k])):
                print(t,': m inf or nan. Stop via []**2 ^o^')
                []**2
            v['s2'][k] = v['m'][k]**2*(np.exp(v['sig2'][k])-1)            
            
            # update the world
            if p['w-dynamic']=='OU':
                v['lam'][k+1] = v['lam'][k] + dt/p['tau_ou']*(p['mu_ou-gm'] - v['lam'][k]) + dW*eta(dim)
            elif p['w-dynamic']=='jump':
                v['lam'][k+1] = v['lam'][k] + dt/p['tau_ou']*(p['mu_ou-gm'] - v['lam'][k]) + dW*np.sign(eta(dim))
            
            # update spikes 
            n_spikes = np.random.binomial(1,p['nu*dt'])
            ix = np.where(n_spikes)[0]
            v['x'][k,ix] += 1 # init cond is zero for entire array
            
            # update samples
            if isinstance(p['sample_method'],str):
                w_var = v['s2'][k,ix] if (p['bayesian'] and np.isnan(p['k'])) else p['k']*v['m'][k,ix]
                if 'gauss' in p['sample_method']:
                    # because of zero init it work
                    v['w'][k,ix] = v['m'][k,ix] + w_var**0.5*eta(len(ix))
                elif 'lognormal' in p['sample_method']:
                    v['w'][k,ix] = np.exp(v['mu'][k,ix] + w_var**0.5*eta(len(ix)))
            else: # take mean
                v['w'][k,ix] = v['m'][k,ix]
            
            # update (true) membrane
            v['u'][k] = v['x'][k,ix].dot(np.exp(v['lam'][k,ix]) - v['w'][k,ix]
                                ) + p['sig0u-gm']*eta(1)
            
            # likelihood
            if 'RL' in p['rule']:
                dmu_like, dsig2_like = v.rl(p,k,t)
            elif 'Binary' in p['rule']:
                dmu_like, dsig2_like = v.binary(p,k,t)
            elif 'Linear' in p['rule']:
                dmu_like, dsig2_like = v.linear(p,k,t)
    
            # Prior: update the synapse (w/o dt)
            dmu_pi, dsig2_pi = [- (v['mu'][k] - p['mu_ou'])/p['tau_ou'], - 2*(v['sig2'][k] - p['sig2_ou'])/p['tau_ou']] if p['bayesian'] else [0,0]
            # final update
            v['mu'][k+1] = v['mu'][k] + (dmu_like + dmu_pi*dt)
            v['sig2'][k+1] = v['sig2'][k] + (dsig2_like + dsig2_pi*dt)

            # for supps
            if p['fig'] == 4:
                # compact (w/o prior)
                supps[0,ix] += np.abs(np.exp(
                                        dmu_like[ix] + dsig2_like[ix]/2 + 
                                        dmu_pi[ix]*dt + dsig2_pi[ix]/2*dt) - 1)
                #supps[0,ix] += np.abs(dmu_like[ix])/v['m'][k,ix]
                supps[1] += v['s2'][k]/v['m'][k]
                supps[2] += n_spikes
    
            # increment
            if mp['online']:
                # update
                # compute error for online estimate
                if t >= p['k_cut']:                
                    err += v.res(k=1)
                                    
                # shift only variables with temporal index back
                for key in v.keys():
                    if key not in ('<sig2u>','m_ou','s2_ou'):
                        v[key][0] = v[key][1]
             
        # end k-loop
        print('fin sim_id {1} after {0}min:'.format(round((time()-t0)/60,1),sim_id))        
        if mp['online']: # normalise      
            tab.loc[sim_id,['MSE','p_in']] = err/(t_num - 1 - p['k_cut'])        
        else: # compute
            tab.loc[sim_id,['MSE','p_in']] = v.res(k=p['k_cut'])   

        # save times series for plotting fig 2
        if fig == 2: # timeseries                        
            # save mean, variance, lambda and firing rate of selected synapses
            v_ts = {}
            k_cut = int(mp['cut_steps']*p['tau_ou']/dt)
            for key in ['lam','w','mu','sig2','m','s2']:
                v_ts[key] = v[key][k_cut:-1:mp['down_sample'],p['plt-dim']]
                #v_ts[key] = v[key][:,p['plt-dim']]
            v_ts['tspan'] = np.arange(0,len(v_ts['w']))*dt*mp['down_sample']
            v_ts['nu*dt'] = p['nu*dt'][p['plt-dim']] # save rate
            save_obj(v_ts,'fig_{1}_rule_{0}'.format(p['rule'],fig),res_path)                        
                                                    
        elif fig == 4: # supps
            # save normalised variance, empirical learning rate, firing rate
            
            # normalise by spike count
            dm_m = supps[0]/supps[2]
            s2_m = supps[1]/(t_num - 1 - p['k_cut'])
            nu = p['nu*dt']
            
            # save output dict
            supp_out = {'dm_m':dm_m,'s2_m':s2_m,'nu':nu}
            save_obj(supp_out,'fig_{2}_rule_{0}_id_{1}'.format(p['rule'],
                                                         sim_id,fig),res_path)

    if mp['load_previous_results'] == False:
        # end learning rate loop
        print('Table of sims finished, the result is:')
        print(tab)
        
        if fig not in (2,4):
            save_obj(tab,'fig_{1}_id_{0}'.format(my_args.i,fig),res_path)

    #### Plotting
    plt_figs(fig=fig,mp=mp,lw=2,fontsize=18,
             plt_path=plt_path,res_path=res_path)
    
