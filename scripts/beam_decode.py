#!/usr/bin/env python
"""
Decode a language model using one GPU.
"""
import numpy as np
import argparse
import sys
import json
import caffe

from evaluate import CaptionScorer
from util import restore_weights

def translate(vocab, blob):
    caption = "";
    w = 0;
    while True:
        next_word = vocab[int(blob[w])]
        if w == 0:
            next_word = next_word.title()
        if w > 0 and next_word != "." and next_word != ",":
            caption += " ";
        if next_word == "\"" or next_word[0] == '"':
            caption += "\\"; # Escape
        caption += next_word;
        w += 1
        if caption[-1] == '.' or w == len(blob):
            break
    return caption
    
def beam_decode(
        model,  # net proto definition
        vocab_file,  # model vocab text file
        weights, # pretrained weights to use
        gpu,  # device id
        outfile, # json output
):

    vocab = []
    with open(vocab_file) as f:
        for word in f:
            vocab.append(word.strip())
    print 'Loaded {:,} words into caption vocab'.format(len(vocab))
    
    caffe.init_log(0, 1)
    caffe.log('Using device %s' % str(gpu))
    caffe.set_device(int(gpu))
    caffe.set_mode_gpu()
    
    net = caffe.Net(model, weights, caffe.TEST)
    print 'Loaded proto {} with weights {}'.format(model,weights)
    net.layers[0].load_dataset()
    
    id_to_caption = {}
    iteration = 0
    while True:
        ending = False
        out = net.forward()
        image_ids = net.blobs['image_id'].data
        captions = net.blobs['caption'].data
        scores = net.blobs['log_prob'].data     
        batch_size = image_ids.shape[0]
        
        if captions.shape[0] == batch_size:
            # Decoding a compact net
            beam_size = captions.shape[2]
            for n in range(batch_size):
                if iteration == 0:
                    print "\nhttp://mscoco.org/explore/?id=%d" % image_ids[n][0]
                for b in range(beam_size):
                    cap = translate(vocab, captions[n][0][b])
                    score = scores[n][0][b]
                    if iteration == 0:
                        print '[%d] %.2f %s' % (b,score,cap)        
        else:
            # Decoding an unrolled net
            beam_size = captions.shape[0] / batch_size
            if iteration == 0:
                print "Beam size: %d" % beam_size
            for n in range(batch_size):
                image_id = int(image_ids[n][0])
                if iteration == 0:
                    print "\nhttp://mscoco.org/explore/?id=%d" % image_id
                for b in range(beam_size):
                    cap = translate(vocab, captions[n*beam_size+b])
                    score = scores[n*beam_size+b]
                    if b == 0:
                        if image_id in id_to_caption:
                            ending = True
                        else:
                            id_to_caption[image_id] = cap
                    if iteration == 0:
                        print '[%d] %.2f %s' % (b,score,cap)
        iteration += 1
        if iteration % 1000 == 0:
          print 'Iteration: %d' % iteration
        if ending:
            break

    output = []
    for image_id in sorted(id_to_caption.keys()):
        output.append({
            'image_id': image_id,
            'caption': id_to_caption[image_id]
        })
    with open(outfile, 'w') as f:
        json.dump(output,f)
    print 'Generated %d outputs, saving to %s' % (len(output),outfile)
    s = CaptionScorer()
    s.score(outfile)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Net proto definition.")
    parser.add_argument("--weights", help="Pretrained weights.")
    parser.add_argument("--gpu", type=int, default=0, help="Device id.")
    parser.add_argument("--vocab", required=True, help="Vocab file.")
    parser.add_argument("--outfile", required=True, help="Output file path.")
    args = parser.parse_args()

    restore_weights(args.weights)
    beam_decode(args.model, args.vocab, args.weights, args.gpu, args.outfile)
