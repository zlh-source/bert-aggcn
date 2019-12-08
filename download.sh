#!/bin/bash

cd dataset
if [ ! -d "bert" ]; then
    mkdir bert
    cd bert; mkdir temp
    cd temp
    echo "==> Downloading pre-trained clinical-bert model..."
    wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1
    echo "==> Unarchiving clinical-bert..."
    tar -xvzf pretrained_bert_tf.tar.gz
    rm pretrained_bert_tf.tar.gz
    cd pretrained_bert_tf
    tar -xvzf biobert_pretrain_output_all_notes_150000.tar.gz
    mv ./biobert_pretrain_output_all_notes_150000 ../..
    cd ../..
    rm -rf temp
fi
echo "==> Done."

