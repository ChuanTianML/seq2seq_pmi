DATA_DIR=/mnt/t-chtian/dataClean/data/not_scored/seq2seq

:<<!
mkdir tmp/pmi_model

rm -rf tmp/pmi_model/src2tgt
mkdir tmp/pmi_model/src2tgt
python -m nmt.nmt \
        --src=titl --tgt=comm \
        --vocab_prefix=$DATA_DIR/embedding/vocab  \
        --embed_prefix=$DATA_DIR/embedding/embed \
        --share_vocab=True \
        --train_prefix=$DATA_DIR/train_data/train \
        --dev_prefix=$DATA_DIR/train_data/dev  \
        --test_prefix=$DATA_DIR/train_data/test \
        --out_dir=tmp/pmi_model/src2tgt \
        --num_train_steps=350000 \
        --steps_per_stats=100 \
        --num_layers=4 \
        --residual=True \
        --encoder_type=gnmt \
        --attention=normed_bahdanau \
        --attention_architecture=gnmt_v2 \
        --num_units=512 \
        --dropout=0.2 \ 
        --num_gpus=2 \
        --decay_scheme=luong10
!

rm -rf tmp/pmi_model/tgt2src
mkdir tmp/pmi_model/tgt2src
CUDA_VISIBLE_DEVICES=1,2,3 python -m nmt.nmt \
        --src=comm --tgt=titl \
        --vocab_prefix=$DATA_DIR/embedding/vocab  \
        --embed_prefix=$DATA_DIR/embedding/embed \
        --share_vocab=True \
        --train_prefix=$DATA_DIR/train_data/train \
        --dev_prefix=$DATA_DIR/train_data/dev  \
        --test_prefix=$DATA_DIR/train_data/test \
        --out_dir=tmp/pmi_model/tgt2src \
        --num_train_steps=350000 \
        --steps_per_stats=100 \
        --num_layers=4 \
        --residual=True \
        --encoder_type=gnmt \
        --attention=normed_bahdanau \
        --attention_architecture=gnmt_v2 \
        --num_units=512 \
        --dropout=0.2 \
        --num_gpus=3 \
        --decay_scheme=luong10
