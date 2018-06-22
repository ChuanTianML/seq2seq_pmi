CORPUS=/home/tiwe/t-chtian/dataClean/data/not_scored/test.txt
OP_DIR=/home/tiwe/t-chtian/dataClean/data/not_scored/seq2seq/results/
OP_FILE=data.pmi
DEBUG=1


python compute_pmi.py -debug $DEBUG -i_path $CORPUS -o_path $OP_DIR$OP_FILE


