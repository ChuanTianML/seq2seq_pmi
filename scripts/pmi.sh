#CORPUS=/mnt/t-chtian/dataClean/data/not_scored/test.txt
CORPUS=/mnt/t-chtian/dataClean/data/not_scored/data.all.txt
OP_DIR=/mnt/t-chtian/dataClean/data/not_scored/seq2seq/results/
OP_FILE=data.pmi
VOCAB=/mnt/t-chtian/dataClean/data/not_scored/remain_words.txt
DEBUG=0


python compute_pmi.py -debug $DEBUG -i_path $CORPUS -o_path $OP_DIR$OP_FILE -vocab_path $VOCAB


