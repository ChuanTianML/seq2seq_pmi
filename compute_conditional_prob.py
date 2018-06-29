# encoding=utf-8

import sys
import argparse
from nmt.pmi import Pmi

# params
args = argparse.ArgumentParser('Input Parameters.')
args.add_argument('-o_dir', type=str, dest='o_dir', help='output directory.')
args.add_argument('-batch_size', type=int, default=128, dest='batch_size', help='batch size.')
args.add_argument('-debug', type=int, default=0, dest='debug', help='run as debugging.')
args.add_argument('-debug_num', type=int, default=128, dest='debug_num', help='corpus lines num when debugging.')
args = args.parse_args()

# function
def proc_one_batch(titls, comms, titl2comm_file, comm2titl_file):
    print('titls num: %d' % len(titls))
    titl2comm_probs = pmi_tool.src2tgt_log_probability_batch(titls, comms)
    titl2comm_file.writelines([('%.4f\n' % p) for p in titl2comm_probs])
    comm2titl_probs = pmi_tool.tgt2src_log_probability_batch(titls, comms)
    comm2titl_file.writelines([('%.4f\n' % p) for p in comm2titl_probs])

# main
pmi_tool = Pmi()
#pmi_tool = None
titl2comm_file = open(args.o_dir+'titl2comm.prob', 'w')
comm2titl_file = open(args.o_dir+'comm2titl.prob', 'w')
titl_file = open(args.o_dir+'titl', 'r')
idx = 0
titls = []
comms = []
for comm in open(args.o_dir+'comm', 'r'):
    titl = titl_file.readline().strip()
    comm = comm.strip()
    titls.append(titl)
    comms.append(comm)
    if len(titls) >= args.batch_size: # proc
        proc_one_batch(titls, comms, titl2comm_file, comm2titl_file)
        titls = []
        comms = []
    idx += 1
    if 0 == idx % 100:
        sys.stdout.write('%d lines processed.\r' % (idx))
        sys.stdout.flush()
    if 1 == args.debug and idx > args.debug_num: break
if 0 < len(titls): # proc 
    proc_one_batch(titls, comms, titl2comm_file, comm2titl_file)

titl2comm_file.close()
comm2titl_file.close()
titl_file.close()
print 'finish.'
