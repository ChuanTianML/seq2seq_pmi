


from nmt.pmi import Pmi


print 'initializing pmi tool...'
pmi_tool = Pmi()
src = 'a b c d e f'
tgt = 'how are you today man ?'
src2tgt_prob = pmi_tool.src2tgt_log_probability(src, tgt)
tgt2src_prob = pmi_tool.tgt2src_log_probability(src, tgt)
print 'src: ' + src
print 'tgt: ' + tgt
print('src2tgt: %.2f' % src2tgt_prob)
print('tgt2src: %.2f' % tgt2src_prob)


