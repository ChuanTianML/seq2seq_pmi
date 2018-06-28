# encoding=utf-8

import sys
import jieba
import argparse
from nmt.pmi import Pmi

# params
args = argparse.ArgumentParser('Input Parameters.')
args.add_argument('-i_path', type=str, dest='i_path', help='corpus file path.')
args.add_argument('-o_path', type=str, dest='o_path', help='output file path.')
args.add_argument('-vocab_path', type=str, dest='vocab_path', help='vocab file path.')
args.add_argument('-lever', type=str, default=0.5, dest='lever', help='lever between src2tgt and tgt2src.')
args.add_argument('-debug', type=int, default=0, dest='debug', help='run as debugging.')
args.add_argument('-debug_num', type=int, default=1000, dest='debug_num', help='corpus lines num when debugging.')
args = args.parse_args()

# functions
def character_cut(word):
    chs = []
    for c in unicode(word, 'utf-8'):
        chs.append(c.encode('utf-8'))
    return chs

def re_cut(sentence, vocab):
    ''' not used.
    '''
    sentence = ''.join(sentence.strip().split())
    ws = jieba.lcut(sentence)
    ws = [w.encode('utf-8') for w in ws]
    ws_cut = []
    for w in ws:
        if w in vocab: ws_cut.append(w)
        else: ws_cut += character_cut(w)
    sentence_cut = ' '.join(ws_cut)
    return sentence_cut


# main
## load vocab
words = open(args.vocab_path, 'r').readlines()
vocab_set = set([w.strip() for w in words])

## process data
pmi_tool = Pmi()
o_file = open(args.o_path, 'w')
idx = 0
for line in open(args.i_path, 'r'):
    titl, comm = line.strip().split('\t')
    #titl = re_cut(titl, vocab_set) # pmi_tool do re-cut
    #comm = re_cut(comm, vocab_set)
    pmi, t2c, c2t = pmi_tool.pmi(titl, comm, args.lever)
    o_file.write('%.4f\t%.4f\t%.4f\t%s\t%s\n' % (pmi, t2c, c2t, titl, comm))
    idx += 1
    if 0 == idx%100:
        sys.stdout.write('%d lines processed.\r' % (idx))
        sys.stdout.flush()
    if 1 == args.debug and idx > args.debug_num: break
o_file.close()
print 'finish.'
















