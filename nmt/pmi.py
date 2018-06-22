#encoding=utf-8

import jieba
import tensorflow as tf
from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils


class Pmi():
    def __init__(self, only_tgt2src = False):
        # the directories of models
        src2tgt_dir = '/home/tiwe/t-chtian/dataClean/neur/nmt/tmp/pmi_model/src2tgt'
        tgt2src_dir = '/home/tiwe/t-chtian/dataClean/neur/nmt/tmp/pmi_model/tgt2src'

        # load model
        self.src2tgt_model, self.src2tgt_sess = self.load_model(src2tgt_dir)
        self.tgt2src_model, self.tgt2src_sess = self.load_model(tgt2src_dir)

    def load_model(self, out_dir):
        
        hparams = utils.load_hparams(out_dir)
        if not hparams: 
            raise IOError("Unable to load hparams from %s" % out_dir)
        
        if not hparams.attention:
            print 'no attention.'
            model_creator = nmt_model.Model
        else:
            if (hparams.encoder_type == "gnmt" or hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
                model_creator = gnmt_model.GNMTModel
            elif hparams.attention_architecture == "standard":
                model_creator = attention_model.AttentionModel
            else:
                raise ValueError("Unknown attention architecture %s" % hparams.attention_architecture)
        
        log_device_placement = hparams.log_device_placement
        model = model_helper.create_prob_model(model_creator, hparams, scope=None) # i dont know the meaning of scope

        config_proto = utils.get_config_proto(
            log_device_placement=hparams.log_device_placement,
            num_intra_threads=hparams.num_intra_threads,
            num_inter_threads=hparams.num_inter_threads)
        
        # to do target_session
        sess = tf.Session(target='', config=config_proto, graph=model.graph) # i dont know the meaning of target
        
        # load model variables
        loaded_model = model.model
        with model.graph.as_default():
            latest_ckpt = tf.train.latest_checkpoint(out_dir)
            if latest_ckpt:
                loaded_model.saver.restore(sess, latest_ckpt)
                sess.run(tf.tables_initializer())
            else:
                raise IOError("Unable to load model from %s" % out_dir)
        
        print 'loaded model from ' + out_dir
        
        return model, sess
        #return loaded_model, sess


    def src2tgt_log_probability(self, src, tgt, re_cut=True):
        """ get the log probability of target sentence given source sentence.
        Args:
            src: source sentence; tgt: target sentence
        Return:
            the probability.
        """
        model = self.src2tgt_model
        sess = self.src2tgt_sess

        if re_cut:
            src = self.preproc(src)
            tgt = self.preproc(tgt)
        iterator_feed_dict = {
            model.src_placeholder: [src],
            model.tgt_placeholder: [tgt],
            model.batch_size_placeholder: 1,
        }
        sess.run(model.iterator.initializer, feed_dict=iterator_feed_dict)

        loss, predict_count, batch_size = model.model.eval(sess)

        return -loss/predict_count


    def tgt2src_log_probability(self, src, tgt, re_cut=True):
        """ get the log probability of source sentence given target sentence.
        Args:
            src: source sentence; tgt: target sentence;
        Return:
            the probability.
        """
        model = self.tgt2src_model
        sess = self.tgt2src_sess

        if re_cut: 
            src = self.preproc(src)
            tgt = self.preproc(tgt)
        iterator_feed_dict = {
            model.src_placeholder: [tgt],
            model.tgt_placeholder: [src],
            model.batch_size_placeholder: 1,
        }
        sess.run(model.iterator.initializer, feed_dict=iterator_feed_dict)

        loss, predict_count, batch_size = model.model.eval(sess)
        
        return -loss/predict_count

    def pmi(self, src, tgt, lever=0.5, re_cut=True):
        """ compute pmi (point mutual information) based on seq2seq method.
        Args:
            src: source sentence.
            tgt: target sentence.
            lambda: the lever parameter between src2tgt and tgt2src
        Return:
            thr pmi (point mutual information)
        """
        src2tgt_log_prob = self.src2tgt_log_probability(src, tgt, re_cut)
        tgt2src_log_prob = self.tgt2src_log_probability(src, tgt, re_cut)
        pmi = lever * src2tgt_log_prob + (1.0-lever) * tgt2src_log_prob
        return pmi, src2tgt_log_prob, tgt2src_log_prob

    def preproc(self, sentence):
        """ re-cut the input sentence
        """
        sentence = ''.join(sentence.strip().split())
        words = jieba.lcut(sentence)
        words = [w.encode('utf-8') for w in words]
        sentence = ' '.join(words)
        return sentence



