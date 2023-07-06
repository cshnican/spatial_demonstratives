import numpy as np
import scipy.optimize
from collections import Counter

"""
numbers for FINN_COUNTS from finnish_freqs.csv
FINN_WORDS = ["täällä", "siellä", "tuolla",
                "tänne", "sinne", "tuonne",
                "täältä", "sieltä", "tuolta"]
"""
# Finnish deictic demonstrative frequencies from Lexiteria https://lexiteria.com/word_frequency_list.html
FINN_COUNTS = {"place":  [232946, 94887, 38402],
            "goal": [109576, 42923, 10006],
            "source": [43016, 17587, 3850],
            "unif": [1, 1, 1]}

# Finnish deictic demonstrative frequencies from WorldLex-Finnish http://www.lexique.org/shiny/openlexicon/
# Twitter Frequency (Column: TwitterFreq)
FINN_COUNTS_WORLDLEX_TWITTER = {"place": [3415, 2803, 559],
                                "goal": [925, 1876, 108],
                                "source": [513, 949, 119],
                                "unif": [1,1,1]}
# Blog Frequency (Column: BlogFreq)
FINN_COUNTS_WORLDLEX_BLOG =    {"place": [7498, 8921, 1686],
                                "goal": [3006, 4555, 340],
                                "source": [1413, 3284, 361],
                                "unif": [1,1,1]}

# Combine two counts
FINN_COUNTS_WORLDLEX = {k: [sum(tup) for tup in zip(FINN_COUNTS_WORLDLEX_TWITTER.get(k, 0), FINN_COUNTS_WORLDLEX_BLOG.get(k, 0))] 
for k in set(FINN_COUNTS_WORLDLEX_TWITTER) | set(FINN_COUNTS_WORLDLEX_BLOG)}


# Finnish deictic demonstrative frequencies from OpenSubtitles-Finnish 2016 https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/fi/fi_full.txt
FINN_COUNTS_OPENSUBTITLES = {"place": [297313, 121392, 45204],
                        "goal": [139150, 54970, 11180],
                        "source": [49141, 21193, 4928],
                        "unif": [1,1,1]}


# Turkish deictic demonstrative frequencies from OpenSubtitles-Turkish 2018 https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/tr/tr_full.txt        
# orada is broader term for "there" and "over there", but "şurada" is more restricted to something you can point at.                
"""
TURK_WORDS = ["burada", "şurada", "orada", 
                "buraya", "şuraya", "oraya",
                "buradan", "şuradan", "oradan"]
"""
TURK_COUNTS_OPENSUBTITLES = {"place": [563006, 15217, 216773],
                        "goal": [262977, 17195, 85374],
                        "source": [112143, 4052, 33163],
                        "unif": [1,1,1]}

#FINN_COUNTS = {"place":  [39, 16, 6],
#            "goal": [18, 7, 1.6],
#            "source": [7, 2, .6],
#            "unif": [1, 1, 1]}

def get_exp_fit(prior, distal_levels):
    """Take exponential fit from Finnish to get distribution over other levels"""
    def func(x, a, b):
        return a*np.exp(-b*x)
    a, b = scipy.optimize.curve_fit(func, range(len(prior)), prior)[0]
    dist = func(range(distal_levels), a, b)
    return dist

def get_exp_prior(distal_levels, prior_spec, prior_source = "lexiteria"):
    """Get the prior distribution over distal levels"""

    if prior_source == "lexiteria": # source 1: Lexiteria
            prior_count = FINN_COUNTS
    elif prior_source == "worldlex":
            prior_count = FINN_COUNTS_WORLDLEX
    elif prior_source == "opensubtitles":
            prior_count = FINN_COUNTS_OPENSUBTITLES
    else:
            prior_count = FINN_COUNTS
    


    x = np.concatenate([get_exp_fit(prior_count[i], distal_levels) for i in prior_spec])
    return x/np.sum(x)
