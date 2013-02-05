import pymongo
from data_storage import Summary
import cPickle as pickle
import sys
import time
from joblib import Parallel,delayed

def memo(f):
    # Memoize function f
    table = {}
    def fmemo(*args):
        # when segment() is called on a new string, add the result of segment(new_string) to table dict
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo
    
def get_spanning_set(summary_id,sim_metric,combine,threshold):
    #print 'summary_id'
    #print summary_id
    # get number of sentences
    sentences = pickle.loads(writingsamples.find_one({'id':summary_id})['summary']).sents
    num_sentences = len(sentences)
    #print 'number of sentences in this sample:'
    #print num_sentences
    contributors = []
    for sent_num in xrange(num_sentences):
        contributors.append(get_sent_spanning_set(summary_id,sim_metric,sent_num+1,sentences[sent_num],combine,threshold))
    return contributors

def get_sent_spanning_set(summary_id,sim_metric,sent_num,sentence,combine,threshold):
    #print 'sent_num'
    #print sent_num
    # get sent_len
    span_start = 0
    span_end = len(sentence) - 1
    #print sentence[span_start],sentence[span_end]
    return f(summary_id,sim_metric,combine,threshold,sent_num,span_start,span_end)

@memo    
def f(sample,sim_metric,combine,threshold,sent_num,span_start,span_end):
    #print 'calling f with these args'
    #print span_start,span_end
    if span_start > span_end:
        #print 'true'
        return []
    else:
        spansets = ((score(sample,sim_metric,combine,threshold,sent_num,span_start,i) + f(sample,sim_metric,combine,threshold,sent_num,i+1,span_end)) for i in xrange(span_start,span_end+1))
        return max(spansets, key=get_summary_score)

def score(sample,sim_metric,combine,threshold,sent_num,start,end):
    #print sim_metric
    #print 'calling score.....'
    #print "sent {2}: score({0},{1})".format(start,end,sent_num)
    ######## DEFINE WHICH DB COLLECTION TO USE DEPENDING ON SIM_METRIC
    if sim_metric == 'uni':
        coll = db.unigram_5thresholds
        assert len([i for i in coll.find({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,
                                        'sent_num':sent_num})])==1, "mongodb did not return a unique value for candidate query"
        #print (start,end)+tuple(coll.find_one({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,'sent_num':sent_num})['uni_overlap_match'])
        return [(start,end)+tuple(coll.find_one({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,'sent_num':sent_num})['uni_overlap_match'])]
    elif sim_metric == 'ro':
        coll = db.ro_5thresholds
        res = [i for i in coll.find({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,
                                        'sent_num':sent_num})]
        assert len(res)==1, "mongodb did not return a unique value for candidate query: {}".format(res)
        #print (start,end)+tuple(coll.find_one({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,'sent_num':sent_num})['ro_match'])
        return [(start,end)+tuple(coll.find_one({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,'sent_num':sent_num})['ro_match'])]
    else:
        coll = db.cos_5thresholds
        res = [i for i in coll.find({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,
                                        'sent_num':sent_num})]
        assert len(res)==1, "mongodb did not return a unique value for candidate query: {}".format(res)
        return [(start,end)+tuple(coll.find_one({'sample':sample,'candidate.start':start,'candidate.end':end,'combine':combine,'threshold':threshold,'sent_num':sent_num})['cos_match'])]
def get_summary_score(spanset):
    #print spanset
    try:
        scus = []
        score = 0
        for sent in spanset:
            for start,end,match,scu,weight in sent:
                if scu not in scus: score += weight
                scus.append(scu)
    except TypeError:
        scus = []
        score = 0
        for start,end,match,scu,weight in spanset:
            if scu not in scus: score += weight
            scus.append(scu)
    #print 'score'
    return score

def store_summ_score(id,sim_metric,combine,threshold,spanset):
    summary_score = get_summary_score(spanset)
    summaryscores.save({'id':id, 'spanset':spanset, sim_metric:{'combine':combine,'threshold':threshold,'score':summary_score}})

def score_samples(chunk,sim_metric,combine,threshold):
    for id in chunk:
        spanset = get_spanning_set(id,sim_metric,combine,threshold)
        #print spanset
        store_summ_score(id,sim_metric,combine,threshold,spanset)

def chunks(list,chunksize):
    for i in xrange(0,len(list),chunksize):
        yield list[i:i+chunksize]
        
if __name__ == '__main__':
    start_time = time.time()
    c = pymongo.Connection()
    db = c.tc_storage
    writingsamples = db.writingsamples
    summaryscores = db.summaryscores

    # get list_of_ids
    list_of_ids = [i['id'] for i in writingsamples.find()]
    # commandline args sim_metric, combine, threshold
    Parallel(n_jobs=2,verbose=1)(delayed(score_samples)(chunk,sys.argv[1],sys.argv[2],float(sys.argv[3])) for chunk in chunks(list_of_ids,5))
    print "total time: {}".format(time.time()-start_time)