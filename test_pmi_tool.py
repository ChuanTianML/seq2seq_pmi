


from nmt.probability import Pmi


print 'initializing pmi tool...'
pmi_tool = Pmi()
src = 'a b c d e'
tgt = 'how are you today ?'
src2tgt_prob = pmi_tool.src2tgt_probability(src, tgt)
tgt2src_prob = pmi_tool.tgt2src_probability(src, tgt)
print 'src: ' + src
print 'tgt: ' + tgt
print('src2tgt: %.2f' % src2tgt_prob)
print('tgt2src: %.2f' % tgt2src_prob)


