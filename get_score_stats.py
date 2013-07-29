import pymongo
import scipy
from scipy.stats import pearsonr,spearmanr,kendalltau
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import itertools
import collections
import operator
import csv

class Experiment:
    def __init__(self, 
                 summary_ids, 
                 combine, 
                 metric, 
                 t, 
                 human_scus, 
                 human_scores, 
                 human_rankings, 
                 human_ordinals):
        self.combine = combine
        self.metric = metric
        self.thres = t
        self.scus = get_scus(summary_ids, combine, metric, t)
        self.scorelist = get_score_list(summary_ids, combine, metric, t)
        self.ordinals = get_ordinals(summary_ids, get_summ_rankings(summary_ids, combine, metric, t))
        self.pearson = get_pearson(np.array(human_scores),np.array(self.scorelist))
        self.jaccard = np.mean(jacc_aggregate(self.scus, human_scus))
        self.spearman = get_spearman(self.ordinals, human_ordinals)
        self.kendallstau = get_tau(self.ordinals, human_ordinals)
        self.scu_ratio = np.mean(compare_scus(self.scus, human_scus))

def get_pearson(score_list1, score_list2):
    return pearsonr(score_list1, score_list2)[0]

def get_jaccard(set1, set2):
    return float(len(set1 & set2)) / (len(set1) + len(set2) - len(set1 & set2))

def jacc_aggregate(scu_dict_1, scu_dict_2):
    return [get_jaccard(set(scu_dict_1[sample]), set(scu_dict_2[sample])) for sample in scu_dict_1]

def compare_scus(scu_dict_1, scu_dict_2):
    '''Return a list of ratios, where each is the ratio for a single summary of the number 
    of SCUs assigned by the algorithm to the number of SCUs assigned by the human annotator'''
    return [float(len(scu_dict_1[sample])) / len(scu_dict_2[sample]) for sample in scu_dict_1]  

def get_spearman(ordinal_list1, ordinal_list2):
    return spearmanr(np.array([ordinal_list1, ordinal_list2]).T)[0]

def get_tau(ordinal_list1, ordinal_list2):
    return kendalltau(ordinal_list1, ordinal_list2)[0]

def get_scus(summary_ids, combine, metric, t):
    scus = dict.fromkeys(summary_ids)
    for id in summary_ids:
        res = [i for i in coll.find({'sample':id,'combine':combine,'metric':metric,'threshold':t})]
        assert len(res) == 1, 'mongodb did not return unique doc: {}'.format(res)
        scus[id] = res[0]['machine']['scus']
    return scus

def get_score_list(summary_ids, combine, metric, threshold):
    # returns tuple of scores of the summaries in the order they are listed in summary_ids
    scores = []
    for id in summary_ids:
        res = [i for i in coll.find({'sample':id, 
                                     'combine':combine, 
                                     'metric':metric, 
                                     'threshold':threshold})]
        assert len(res) == 1, 'mongodb did not return unique doc: {}'.format(res)
        scores.append(res[0]['machine']['weightedsum'])
    return scores

def get_summ_rankings(summary_ids, combine, metric, threshold):
    rankings = []
    for id in summary_ids:
        res = [i for i in coll.find({'sample':id, 'combine':combine, 'metric':metric, 'threshold':threshold})]
        assert len(res) == 1, 'mongodb did not return unique doc: {}'.format(res)
        score = res[0]['machine']['weightedsum']
        rankings.append((score,id))
    return sorted(rankings,reverse=True)

def get_ordinals(summary_ids, rankings):
    '''Return tuple of ordinals for each summary id in the order of `summary_ids` input.
    Accepts:
    summary_ids: list of ids of the student summaries
    rankings: list of (score, id) tuples
    '''
    scores = [score for score, summ_id in rankings]
    ids = [summ_id for score, summ_id in rankings]
    ranks = [(ids.index(summ_id) + 1) for summ_id in ids]
    observations = set(scores)
    if len(scores)!=len(observations):
        counter = collections.Counter(scores) 
	dup_ordinals = [get_dup_ordinals(obs, scores) for obs, count in counter.items() if count > 1]
        #replace values in ordinals with mean(ordinals)
	for r in ranks:
            if r in itertools.chain.from_iterable((j for j in dup_ordinals)):
		r_index = ranks.index(r)
		ranks[r_index] = np.mean([i for i in itertools.ifilter(lambda x: r in x, dup_ordinals)])
    return ranks

def get_dup_ordinals(obs, scores):
    return [(ind + 1) for ind, observation in enumerate(scores) if observation == obs]

def stat_sort(stat):
    """Present results ordered by metric and combine function."""
    return sorted(stat, key = operator.itemgetter(2,1,3))

def get_human_attr(summary_ids):
    human_scores = []
    human_scus = dict.fromkeys(summary_ids)
    for id in summary_ids:
        human_scus[id] = coll.find_one({'sample':id})['human']['scus']
        score = coll.find_one({'sample':id})['human']['weighted_sum']
        human_scores.append(score)
    human_rankings = sorted((i for i in itertools.izip(human_scores, summary_ids)), reverse=True)
    human_ordinals = get_ordinals(summary_ids, human_rankings)
    return (human_scus, human_scores, human_rankings, human_ordinals)

def get_exp_attr(metrics, 
                 experiments, 
                 summary_ids, 
                 human_scus, 
                 human_scores, 
                 human_rankings, 
                 human_ordinals):
	pearson = []
	jaccard = []
	spearman = []
	kendallstau = []
	scu_ratio = []
	for combine in experiments:
		experiments[combine] = dict.fromkeys(metrics)
		for metric in experiments[combine]:
			experiments[combine][metric] = dict.fromkeys(str(i) for i in db.experiments.find_one({'combine':combine,'metric':metric})['thresholds'])
			for t in experiments[combine][metric]:
				trial = Experiment(summary_ids,combine,metric,float(t),human_scus,human_scores,human_rankings,human_ordinals)
				experiments[combine][metric][t] = trial
				pearson.append((trial.pearson,trial.combine,trial.metric,trial.thres))
				jaccard.append((trial.jaccard,trial.combine,trial.metric,trial.thres))
				spearman.append((trial.spearman,trial.combine,trial.metric,trial.thres))
				kendallstau.append((trial.kendallstau,trial.combine,trial.metric,trial.thres))
				scu_ratio.append((trial.scu_ratio,trial.combine,trial.metric,trial.thres))
	return [pearson,spearman,kendallstau,jaccard,scu_ratio]

def write_raw_data(data,excluded):
	with open('experiment_results/rawstats_excluding_'+excluded,'wb') as f:
		csvwriter = csv.writer(f)
		csvwriter.writerows(data)

def write_exp_rankings(data,excluded):
	with open('experiment_results/experiment_rankings_excluding_'+excluded,'wb') as f:
		csvwriter = csv.writer(f)
		csvwriter.writerows(data)

def store_in_db(excluded,stat_names,labels,results):
	for crossval in results:
		doc = dict(itertools.izip(['excluded','metric','combine','thres']+stat_names,(excluded,)+crossval))
		db.experimentstats.save(doc)

def cross_val(summary_ids,stat_names,metrics,experiments,labels):
	for i in xrange(len(summary_ids)):
		excluded = summary_ids[i]
		human_scus,human_scores,human_rankings,human_ordinals = get_human_attr(summary_ids[:i]+summary_ids[i+1:])
		statistics = get_exp_attr(metrics,experiments,summary_ids[:i]+summary_ids[i+1:],human_scus,human_scores,human_rankings,human_ordinals)
		sorted_stats = []
		for stat in statistics:
			sorted_stats.append([score for (score,c,m,t) in stat_sort(stat)])
		results1 = [[m for (m,c,t) in labels]]+[[c for (m,c,t) in labels]]+[[t for (m,c,t) in labels]]+sorted_stats
		header1 = ['metric','combine','threshold',stat_names[0],stat_names[1],stat_names[2],stat_names[3],stat_names[4]]
	#write_raw_data(itertools.chain(iter([header1]),itertools.izip(*results1)),excluded)
		store_in_db(excluded,stat_names,labels,itertools.izip(*results1))

def get_all_confint(labels):
	for m,c,t in labels:
		pearson = []
		spearman = []
		jaccard = []
		kendallstau = []
		scu_ratio = []
		for res in db.experimentstats.find({'metric':m,'combine':c,'thres':t}):
			pearson.append(res['pearson'])
			spearman.append(res['spearman'])
			jaccard.append(res['jaccard'])
			kendallstau.append(res['kendallstau'])
			scu_ratio.append(res['scu_ratio'])
		stats = [pearson,spearman,jaccard,kendallstau,scu_ratio]
		statmeans = [np.mean(stat) for stat in stats]
		intervals = []
		statdevs = [np.std(stat) for stat in stats]
		for mean,std in itertools.izip(statmeans,statdevs):
			intervals.append(compute_confint(mean,std))
		doc=dict(itertools.izip(['excluded','metric','combine','thres','pearson_int','spearman_int','jaccard_int','kendint','ratio_int'],['none',m,c,t]+intervals))
		db.experimentstats.save(doc)

def compute_confint(mean,std):
	# assumes gaussian distribution and returns (a,b) = (percent point function at 0.025, percent point function at 0.975)
	return scipy.stats.norm.interval(alpha=0.95,loc=mean,scale=std)

def store_ranking_interval():
	'''
	returns the lowest and highest possible ranking for each experiment for the Pearson,
	Spearman, Kendall's tau and Jaccard set similarity statistics
	'''
	pass

def plot(n, markers, axes, stat, x, name, yticks):
    centers = [center for center,metric,combine,thres,interval in stat][:n]
    metrics = [metric for center,metric,combine,thres,interval in stat][:n]
    yerrs = [(interval[1]-interval[0])/float(2) for center,metric,combine,thres,interval in stat][:n]
    for rank, center, metric, error in itertools.izip(x, centers, metrics, yerrs):
	axes.errorbar(rank, center, error, marker=markers[metric], mec='black', mfc='black', ecolor='black')
    title_fontprop = FontProperties(family='times new roman', size='medium')
    axes.set_title(name, fontproperties=title_fontprop)
    fontprop = FontProperties(family='times new roman', size='large')
    axes.set_ylabel('parameter score', fontproperties=fontprop)
    axes.set_yticks(yticks)
    axes.set_yticklabels([str(tick) for tick in yticks], family='times new roman')
	
def put_legend_on_axes(axes, markers, stat, x):
    centers = [center for center, metric, combine, thres, interval in stat][:n]
    metrics = [metric for center, metric, combine, thres, interval in stat][:n]
    yerrs = [(interval[1] - interval[0]) / float(2) for center, metric, combine, thres, interval in stat][:n]
    mfc = 'black'
    all_metrics = {'cos':'LVc','ro':'RO','uni':'UNI'}
    for mkey in all_metrics:
        if list(x[np.where(np.array(metrics)==mkey)]):
            add_metric_to_legend(axes,markers[mkey],mfc,all_metrics[mkey],x[np.where(np.array(metrics)==mkey)],np.array(centers)[np.where(np.array(metrics)==mkey)],np.array(yerrs)[np.where(np.array(metrics)==mkey)])
    fontprop = FontProperties(family='times new roman',size='medium')
    axes.legend(loc='upper center',bbox_to_anchor=(0.5, 1.5),ncol=4,prop=fontprop)

def add_metric_to_legend(axes,marker,mfc,metric,x_data,y_data,y_err_data):
    points = [i for i in itertools.izip(x_data,y_data,y_err_data)]
    axes.errorbar(points[0][0],points[0][1],points[0][2],marker=marker,mec='black',mfc=mfc,ecolor='black',label=metric)

def rank_experiments(labels):
	# get centers
	pearson = []
	spearman = []
	kendall = []
	jacc = []
	for m,c,t in labels:
		p = [i for i in db.experimentstats.find({'metric':m,'combine':c,'thres':t,'pearson_int':{'$exists':True}})]
		assert len(p) == 1, 'mongodb did not return unique result for {0},{1},{2}, pearsonint'.format(m,c,t)
		pcenter = np.mean(p[0]['pearson_int'])
		pearson.append((pcenter,m,c,t,tuple(p[0]['pearson_int'])))
		
		s = [i for i in db.experimentstats.find({'metric':m,'combine':c,'thres':t,'spearman_int':{'$exists':True}})]
		assert len(s) == 1, 'mongodb did not return unique result for {0},{1},{2}, spearmanint'.format(m,c,t)
		scenter = np.mean(s[0]['spearman_int'])
		spearman.append((scenter,m,c,t,tuple(s[0]['spearman_int'])))
		
		k = [i for i in db.experimentstats.find({'metric':m,'combine':c,'thres':t,'kendint':{'$exists':True}})]
		assert len(k) == 1, 'mongodb did not return unique result for {0},{1},{2}, kendint'.format(m,c,t)
		kcenter = np.mean(k[0]['kendint'])
		kendall.append((kcenter,m,c,t,tuple(k[0]['kendint'])))
					   
		j = [i for i in db.experimentstats.find({'metric':m,'combine':c,'thres':t,'jaccard_int':{'$exists':True}})]
		assert len(j) == 1, 'mongodb did not return unique result for {0},{1},{2}, jacc_int'.format(m,c,t)
		jcenter = np.mean(j[0]['jaccard_int'])
		jacc.append((jcenter,m,c,t,tuple(j[0]['jaccard_int'])))

	stats = [pearson,spearman,kendall,jacc]
	names = ['pearson','spearman','kendalltau','jaccard']
	res = []
	for i in xrange(len(stats)):
		res.append(sorted(stats[i],reverse=True))
		stats[i] = ([(c,m,t,interval) for center,c,m,t,interval in sorted(stats[i],reverse=True)])
	for stat,name in itertools.izip(stats,names):
		with open("/Users/EmilyChen/Dropbox/pyramids/doc/exp_rankings/experiment_rankings_with_intervals_{}".format(name), 'wb') as f:
		    writer = csv.writer(f)
		    writer.writerows(stat)
	return res

if __name__ == '__main__':
	plt.ion()
	c = pymongo.Connection()
	db = c.tc_storage
	coll = db.writingsamplestest
    # generate an immutable, ordered tuple of summary ids to be the reference for generating scores for each of the experiments
	summary_ids = tuple(coll.distinct('sample'))
		
	combine_functions = ['max','mean','min']
	metrics = ['uni','ro','cos']
	experiments = dict.fromkeys(combine_functions)
	stat_names = ['pearson','spearman','kendallstau','jaccard','scu_ratio']
	# evaluate results for dummy data to generate labels
	human_scus,human_scores,human_rankings,human_ordinals = get_human_attr(summary_ids)
	labels = [(m,c,t) for (score,c,m,t) in stat_sort(get_exp_attr(metrics,experiments,summary_ids,human_scus,human_scores,human_rankings,human_ordinals)[0])]
#cross_val(summary_ids,stat_names,metrics,experiments,labels)
#get_all_confint(labels)
	all_experiment_rankings = rank_experiments(labels)
	fig,axes = plt.subplots(nrows=3, ncols=1, sharex=True)
	n = 10
	axes[2,].set_xticklabels(np.arange(0,n+2,2),family='times new roman')
	fontprop = FontProperties(family='times new roman',size='large')
	axes[2,].set_xlabel('experiment ranking',fontproperties=fontprop)
	axes[0,].set_xlim(0,n+1)
	axes[0,].set_ylim(0.8,1)
	axes[1,].set_ylim(0.8,1)
	axes[2,].set_ylim(0.7,1)
	#axes[3,].set_ylim(0.4,0.6)
	yticks = {0:np.array(xrange(8,11))*0.1,1:np.array(xrange(8,11))*0.1,2:np.array(xrange(7,11))*0.1,3:np.array(xrange(4,7))*0.1}
	x = np.arange(1,n+1)
	i = 0
	plot_titles = ["(a) Pearson", "(b) Spearman", "(c) Kendall's tau"]
        markers = {'cos':'x', 'ro':'<', 'uni':'s'}
        for stat in all_experiment_rankings[:-1]:
            plot(n, markers, axes[i,], stat, x, plot_titles[i], yticks[i])
            i+=1
	put_legend_on_axes(axes[0,], markers, all_experiment_rankings[0], x)
	fig.set_size_inches(10,15)
	fig.subplots_adjust(hspace=0.3)
	plt.draw()
