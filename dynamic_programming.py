"""Usage: python dynamic_programming.py [sim_metric] [combine] [threshold]"""
import argparse
import pymongo
import cPickle as pickle
import sys

from data_storage import Summary
from joblib import Parallel,delayed

def memo(f):
    # Memoize function f
    table = {}
    def fmemo(*args): 
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo
    
def get_spanning_set(summary_id, sim_metric, combine, threshold):
    sentences = pickle.loads(writingsamples.find_one({'id':summary_id})['summary']).sents
    num_sentences = len(sentences)
    contributors = []
    for sent_num in xrange(num_sentences):
        contributors.append(get_sent_spanning_set(summary_id, sim_metric, sent_num+1, sentences[sent_num], combine, threshold))
    return contributors

def get_sent_spanning_set(summary_id, sim_metric, sent_num, sentence, combine, threshold):
    span_start = 0
    span_end = len(sentence) - 1
    return f(summary_id,sim_metric,combine,threshold,sent_num,span_start,span_end)

@memo    
def f(sample,sim_metric,combine,threshold,sent_num,span_start,span_end):
    if span_start > span_end:
        return []
    else:
        spansets = ((score(sample, sim_metric, combine, threshold, sent_num, span_start, i) + f(sample, sim_metric, combine, threshold, sent_num, i+1, span_end)) for i in xrange(span_start, span_end+1))
        return max(spansets, key=get_summary_score)

def score(sample, sim_metric, combine, threshold, sent_num, start, end):
    similarity_metric_params = {"uni": {"collection": db.unigram_5thresholds, 
                                "db_search_key": "uni_overlap_match"}, 
                                "ro":, {"collection": db.ro_5thresholds, 
                                "db_search_key": "ro_match"}, 
                                "cos": {"collection": db.cos_5thresholds, 
                                "db_search_key": "cos_match"}}
    coll = similarity_metric_params[sim_metric]["collection"]
    db_search_key = similarity_metric_params[sim_metric]["db_search_key"]
    assert len([i for i in coll.find({"sample":sample, "candidate.start":start, "candidate.end":end, "combine":combine, "threshold":threshold, "sent_num":sent_num})])==1, "mongodb did not return a unique value for candidate query"
    return [(start, end) + tuple(coll.find_one({"sample":sample, "candidate.start":start, "candidate.end":end, "combine":combine, "threshold":threshold, "sent_num":sent_num})[db_search_key]

def get_summary_score(spanset):
    scus = []
    score = 0
    try: 
        for sent in spanset:
            for start, end, match, scu, weight in sent:
                if scu not in scus: score += weight
                scus.append(scu)
    except TypeError:
       for start, end, match, scu, weight in spanset:
            if scu not in scus: score += weight
            scus.append(scu)
    return score

def store_summ_score(summ_id, sim_metric, combine, threshold, spanset):
    summary_score = get_summary_score(spanset)
    summaryscores.save({'id':summ_id, 'spanset':spanset, sim_metric:{'combine':combine, 'threshold':threshold, 'score':summary_score}})

def score_samples(chunk, sim_metric, combine, threshold):
    for summ_id in chunk:
        spanset = get_spanning_set(summ_id, sim_metric, combine, threshold)
        store_summ_score(summ_id, sim_metric, combine ,threshold, spanset)

def chunks(list, chunksize):
    for i in xrange(0, len(list), chunksize):
        yield list[i:i+chunksize]
        
if __name__ == '__main__':
    c = pymongo.Connection()
    db = c.tc_storage
    writingsamples = db.writingsamples
    summaryscores = db.summaryscores
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", help="The method used to compute a similarity score for two substrings")
    parser.add_argument("combine", help="The function used to select one similarity score from the set generated for a single SCU. Has one of the following values: [min, mean, max]")
    parser.add_argument("threshold", help="The similarity score above which two substrings are said to be a match", type=float)
    args = parser.parse_args() 
    list_of_ids = [i['id'] for i in writingsamples.find()]
    Parallel(n_jobs=2, verbose=1)(delayed(score_samples)(args.metric, args.combine, args.threshold) for chunk in chunks(list_of_ids, 5))
