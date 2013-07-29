Automated Pyramid Scoring of Summaries Using Distributional Semantics
==============================

The pyramid method of summarization evaluation [(Passonneau, 2004)](http://acl.ldc.upenn.edu/hlt-naacl2004/main/pdf/91_Paper.pdf) uses the distribution of content over a pool of human summaries to identify summarization content units (SCUs) that are assigned weights based on their frequency in the corpus of summaries.

This study extended the dynamic programming approach used in [(Harnly, 2005)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.59.8121) to automate summarization evaluation with the pyramid method. To test another method of representing the semantic profile of content units, we incorporated a latent variable model [(Guo and Diab, 2012)](https://www.aclweb.org/anthology-new/P/P12/P12-1091v2.pdf) in the dynamic programming algorithm and found excellent performance for matching the absolute scores of individual summaries and improved performance for identifying the content units included in each summary.

These scripts are the key components of our implementation of the work described in the paper.

Data
----

5 reference summaries were used to create a pyramid by human annotation and 20 target summaries were evaluated by the pyramid method. We compared results for human and automated evaluation systems.


Files
------

* determine_thresholds.py outputs 5 threshold values to set as the minimum semantic similarity score that the algorithm will use to classify a match between a string in the target summary being evaluated and a SCU in the pyramid. These 5 values are the semantic similarity scores of strings in the target summaries that have been annotated by a human as semantically equivalent to an SCU in the pyramid, that have inverse cumulative distribution function values of 0.05, 0.10, 0.15, 0.20, 0.25. This script uses a gaussian kernel density estimator to estimate a continuous distribution of similarity scores and plots the distribution.

* dynamic_programming.py runs the dynamic programming algorithm to return the sum of the SCU weights for each of the target summaries.

* get_score_stats.py computes Pearson's correlation of the absolute score for individual summaries, Spearman's rank-order correlation of the ranking of the summaries, Kendall's tau (another metric for summary rank-order correlation) and Jaccard similarity of retrieved content unit sets. This script generates a plot to show the performance of 60 different systems for each of these metrics, displaying a system's performance with a 95% confidence interval.

Contact
-------
* Rebecca J. Passonneau, Columbia University Center for Computational Learning Systems [(becky@ccls.columbia.edu)](becky@ccls.columbia.edu)
* Emily Chen, Columbia University Department of Computer Science [(ec2805@columbia.edu)](ec2805@columbia.edu)
* Weiwei Guo, Columbia University Department of Computer Science [(weiwei@cs.columbia.edu)](weiwei@cs.columbia.edu)
* Dolores Perin, Columbia University Teachers College [(perin@tc.edu)](perin@tc.edu)