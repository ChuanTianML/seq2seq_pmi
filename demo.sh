mkdir tmp/nmt_model

:<<!
#rm -r tmp/nmt_model/src2tgt
mkdir tmp/nmt_model/src2tgt
python -m nmt.nmt \
        --src=vi --tgt=en \
        --vocab_prefix=tmp/nmt_data/vocab  \
        --train_prefix=tmp/nmt_data/train \
        --dev_prefix=tmp/nmt_data/tst2012  \
        --test_prefix=tmp/nmt_data/tst2013 \
        --out_dir=tmp/nmt_model/src2tgt \
        --num_train_steps=500 \
        --steps_per_stats=10 \
        --num_layers=2 \
        --num_units=64 \
        --dropout=0.2 
        --num_gpus=2
!

rm -r tmp/nmt_model/tgt2src
mkdir tmp/nmt_model/tgt2src
python -m nmt.nmt \
        --src=en --tgt=vi \
        --vocab_prefix=tmp/nmt_data/vocab  \
        --train_prefix=tmp/nmt_data/train \
        --dev_prefix=tmp/nmt_data/tst2012  \
        --test_prefix=tmp/nmt_data/tst2013 \
        --out_dir=tmp/nmt_model/tgt2src \
        --num_train_steps=10000 \
        --steps_per_stats=10 \
        --num_layers=2 \
        --num_units=64 \
        --dropout=0.2 
        --num_gpus=2
