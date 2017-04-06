"""
BUT-rule feature extractor

"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time

warnings.filterwarnings("ignore")


def text_first(text, part):
    if part in text:
        return ''.join(text.split(part)[1:])
    else:
        return ''


def extract_features(revs):
    but_fea = []
    but_ind = []
    but_fea_cnt = 0
    implicit_negation_fea_before = []
    implicit_negation_fea = []
    implicit_negation_ind = []
    implicit_negation_fea_cnt = 0
    for rev in revs:
        text = rev["text"]
        implicit_negation_ind.append(0)
        fea = ''
        fea_before = ''
        if ' but ' in text:
            but_ind.append(1)
            # make the text after 'but' as the feature
            fea = text.split('but')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        elif ' however ' in text:
            but_ind.append(1)
            # make the text after 'but' as the feature
            fea = text.split('but')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        elif ' unfortunately ' in text:
            but_ind.append(1)
            # make the text after 'but' as the feature
            fea = text.split('but')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            but_ind.append(0)
            # only if not but already
            if " avoid" in text:
                implicit_negation_ind.append(1)
                # make the text after 'but' as the feature
                fea = text.split(' avoid')[1].split('.')[0].split(',')[0]
                fea = fea.strip().replace('  ', ' ')
                implicit_negation_fea_cnt += 1
            elif " hardly" in text:
                implicit_negation_ind.append(1)
                # make the text after 'but' as the feature
                fea = text.split(' hardly')[1].split('.')[0].split(',')[0]
                fea = ''.join(fea)
                fea = fea.strip().replace('  ', ' ')
                implicit_negation_fea_cnt += 1
        but_fea.append(fea)
        implicit_negation_fea.append(fea)
        implicit_negation_fea_before.append(fea_before)

    print '#but %d' % but_fea_cnt
    print '#implicit negation: %d' % implicit_negation_fea_cnt
    return {
        'but_text': but_fea,
        'but_ind': but_ind,
        'implicit_negation_before_text': implicit_negation_fea_before,
        'implicit_negation_text': implicit_negation_fea,
        'implicit_negation_ind': implicit_negation_ind,
    }


if __name__ == "__main__":
    data_file = sys.argv[1]
    print "loading data..."
    x = cPickle.load(open(data_file, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    """
                    len     description
    revs            79654   reviews,
    W               17238   embedding (first is zero, not used)
    W2              17238   Random vecs (first is zero, not used)
    word_idx_map    17237   word to index({'unimaginative': 1})
    vocab           17237   number of occurence per word ({'and': 20807.0})
    """
    print "data loaded!"
    fea = extract_features(revs)
    cPickle.dump(fea, open("%s.fea.p" % data_file, "wb"))
    print "feature dumped!"
