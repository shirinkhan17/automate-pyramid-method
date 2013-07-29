import pymongo
import sys
import numpy as np
import itertools
import collections
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from scipy import stats
from matplotlib import pyplot as plt
from decimal import Decimal 

def ppf_discrete(corrects,a):
    mass_densities = np.array(np.bincount(corrects),dtype=float)/np.bincount(corrects).sum()
    cumulative_densities = mass_densities.cumsum()/mass_densities.sum()
    i = 0
    for element in cumulative_densities:
        if element >= a:
            return range(max(sample))[i]
        else:
            i+=1

def ppf_kernelest(est,corrects,a):
    A = est.integrate_box_1d(min(corrects),max(corrects))
    print 'This is the total mass of the scores in your sample'
    print A
    greaterthan = []
    for score in corrects:
        if est.integrate_box_1d(0,score)>=a:
            greaterthan.append(score)
    return min(greaterthan)
    
def get_gaussian_kde(corrects):
    try:
        est=stats.gaussian_kde(corrects)
        p_values = [0.05,0.1,0.15,0.2,0.25]
        thresholds = dict.fromkeys(str(i) for i in p_values)
        for a in p_values:
            thresholds[str(a)] = ppf_kernelest(est,corrects,a)
        return (est,thresholds)
    except ValueError as e:
        print e, corrects

def plot_compare(results):
    # make a subplot for each of the combine functions
    x = np.linspace(0,1,600)
    fig, ax = plt.subplots(3,1,sharex=True,sharey=True)
    fig.set_size_inches(3.1,4.66)
    plt.subplots_adjust(top=0.81)
    tickfont = matplotlib.font_manager.FontProperties(family='times new roman',style='normal',size=8,weight='normal')
    titlefont = matplotlib.font_manager.FontProperties(family='times new roman',style='normal',size=10,weight='normal')
    i = 0
    labelfont = FontProperties(family='times new roman',size=8)
    for combine in results:
        y1 = results[combine]['uni'].evaluate(x)
        L1 = Line2D(x,y1,linestyle='-.')
        ax[i].plot(x,y1,'-.k')
        y2 = results[combine]['ro'].evaluate(x)
        L2 = Line2D(x,y2,linestyle=':')
        ax[i].plot(x,y2,':k')
        y3 = results[combine]['control'].evaluate(x)
        L3 = Line2D(x,y3,linestyle='--')
        ax[i].plot(x,y3,'--k')
        ax[i].set_ylim(0.0,3.0)
        for label in ax[i].get_xticklabels()+ax[i].get_yticklabels():
            label.set_fontproperties(tickfont)
        title = ax[i].set_title(combine)
        title.set_fontproperties(titlefont)
        ax[i].set_ylabel('density',fontproperties=labelfont)
        i+=1
    ax[i-1].set_xlabel('pairwise similarity score',fontproperties=labelfont)
    legend = fig.legend([L1,L2,L3],['Unigram Overlap','Ratcliff/Oberschelp','Latent Vector Cosine Similarity'],prop={'family':'times new roman','size':8},labelspacing = 0.3,loc=(0.22,0.86))
    fig.savefig('thresholds_with_labels')

def prettyprint(combine,metric,thresholds):
    with open('all_thresholds','a') as f:
        f.write(combine+'\t'+metric+'\n')
        for t in sorted(thresholds.items()):
            f.write('inverse cdf = {0}, similarity score = {1}\n'.format(t[0],t[1]))
        f.write('thresholds: {}\n\n'.format(sorted([Decimal(t).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) for t in thresholds.values()])))
    
if __name__ == '__main__':
    c = pymongo.Connection()
    db = c.tc_storage
    metrics = ['uni','ro','control']
    combine_functions = ['max','mean','min']
    results = dict.fromkeys(combine_functions)
    for key in results:
        results[key] = dict.fromkeys(metrics)
    for combine,metric in itertools.product(combine_functions,metrics):
        if metric in ['control','augmented']:
            corrects = np.array([i['scores'][combine]['cos'][metric] for i in db.correct_all4.find()])
        else:
            corrects = np.array([i['scores'][combine][metric] for i in db.correct_all4.find()])
        kernel_dens_est,thresholds = get_gaussian_kde(corrects)
        prettyprint(combine,metric,thresholds)
        results[combine][metric] = kernel_dens_est
    plot_compare(results)    
