DATA_DIR=/home/tiwe/t-chtian/dataClean/data/not_scored/seq2seq

mkdir tmp/pmi_model

rm -r tmp/pmi_model/src2tgt
mkdir tmp/pmi_model/src2tgt
python -m nmt.nmt \
        --src=titl --tgt=comm \
        --vocab_prefix=$DATA_DIR/vocab  \
        --share_vocab=True \
        --train_prefix=$DATA_DIR/train \
        --dev_prefix=$DATA_DIR/dev  \
        --test_prefix=$DATA_DIR/test \
        --out_dir=tmp/pmi_model/src2tgt \
        --num_train_steps=200000 \
        --steps_per_stats=100 \
        --num_layers=4 \
        --residual=True \
        --encoder_type=gnmt \
        --attention=normed_bahdanau \
        --attention_architecture=gnmt_v2 \
        --num_units=128 \
        --dropout=0.2 \ 
        --num_gpus=2

:<<!
rm -r tmp/pmi_model/tgt2src
mkdir tmp/pmi_model/tgt2src
python -m nmt.nmt \
        --src=comm --tgt=titl \
        --vocab_prefix=$DATA_DIR/vocab  \
        --share_vocab=True \
        --train_prefix=$DATA_DIR/train \
        --dev_prefix=$DATA_DIR/dev  \
        --test_prefix=$DATA_DIR/test \
        --out_dir=tmp/pmi_model/tgt2src \
        --num_train_steps=200000 \
        --steps_per_stats=100 \
        --num_layers=4 \
        --residual=True \
        --encoder_type=gnmt \
        --attention=normed_bahdanau \
        --attention_architecture=gnmt_v2 \
        --num_units=128 \
        --dropout=0.2 \
        --num_gpus=2
!       
