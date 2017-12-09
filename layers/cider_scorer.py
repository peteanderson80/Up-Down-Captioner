#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

# Modified to be more amenable to efficiently scoring minibatches during RNN training.


from collections import defaultdict
import numpy as np
import math

from scripts.preprocess_coco import *


def precook(words, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    counts = defaultdict(int)
    for k in xrange(1,n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(split_sentence(ref), n) for ref in refs]

def cook_refs_eos(refs, n=4):
    return [precook(split_sentence(ref)+['.'], n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)


class CiderScorer(object):
    """CIDEr scorer.
    """

    def __init__(self, gt_paths, n=4, sigma=6.0, include_eos=False):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        # process reference captions        
        self.crefs = {}
        for gt_path in gt_paths:
          coco = COCO(gt_path)
          for imgId in coco.getImgIds():
              assert imgId not in self.crefs
              refs = [item['caption'] for item in coco.imgToAnns[imgId]]
              if include_eos:
                  self.crefs[imgId] = cook_refs_eos(refs)
              else:
                  self.crefs[imgId] = cook_refs(refs)
        # compute idf
        self.document_frequency = defaultdict(float)
        for refs in self.crefs.values():
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
                self.document_frequency[ngram] += 1
        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))


    def compute_scores(self, image_ids, captions):

        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.iteritems():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].iteritems():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        scores = []
        for imgId, test in zip(image_ids,captions):
            refs = self.crefs[imgId]
            test = cook_test(test)
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return np.array(scores)

if __name__ == "__main__":

    gt_paths = ['data/coco/captions_val2014.json']
    cider = CiderScorer(gt_paths)
    # Inputs should be lower case, tokenized, without full stop (to match training tokenization)
    captions = [
      ['a', 'brown', 'teddy', 'bear', 'sitting', 'in', 'a', 'basket'],
      ['a', 'motorcycle', 'parked', 'on', 'the', 'side', 'of', 'a', 'road'],
      ['a', 'dog', 'sitting', 'on', 'a', 'bench', 'in', 'a', 'city']
    ]
    image_ids = [42,73,74]
    scores = cider.compute_scores(image_ids, captions)
    np.testing.assert_approx_equal(scores[0], 0.087433, significant=4)
    np.testing.assert_approx_equal(scores[1], 1.0032, significant=4)
    np.testing.assert_approx_equal(scores[2], 0.4705, significant=4)
    print scores





