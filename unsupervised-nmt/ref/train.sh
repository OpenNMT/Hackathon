#! /bin/sh

model_dir=unsupervised-nmt-enfr

src_vocab=data/unsupervised-nmt-enfr/en-vocab.txt
tgt_vocab=data/unsupervised-nmt-enfr/fr-vocab.txt
src_emb=data/unsupervised-nmt-enfr/wmt14m.en300.vec
tgt_emb=data/unsupervised-nmt-enfr/wmt14m.fr300.vec

src=data/unsupervised-nmt-enfr/train.en
tgt=data/unsupervised-nmt-enfr/train.fr
src_trans=data/unsupervised-nmt-enfr/train.en.m1
tgt_trans=data/unsupervised-nmt-enfr/train.fr.m1

src_test=data/unsupervised-nmt-enfr/newstest2014.en.tok
tgt_test=data/unsupervised-nmt-enfr/newstest2014.fr.tok
src_test_trans=data/unsupervised-nmt-enfr/newstest2014.en.tok.m1
tgt_test_trans=data/unsupervised-nmt-enfr/newstest2014.fr.tok.m1

timestamp=$(date +%s)
score_file=scores-${timestamp}.txt

> ${score_file}

score_test()
{
    echo ${src_test_trans} >> ${score_file}
    perl multi-bleu.perl ${tgt_test} < ${src_test_trans} >> ${score_file}
    echo ${tgt_test_trans} >> ${score_file}
    perl multi-bleu.perl ${src_test} < ${tgt_test_trans} >> ${score_file}
}

score_test

for i in $(seq 2 5); do
    # Train for one epoch.
    python ref/training.py \
           --model_dir ${model_dir} \
           --src ${src} \
           --tgt ${tgt} \
           --src_trans ${src_trans} \
           --tgt_trans ${tgt_trans} \
           --src_vocab ${src_vocab} \
           --tgt_vocab ${tgt_vocab} \
           --src_emb ${src_emb} \
           --tgt_emb ${tgt_emb}

    # Evaluate on test files.
    src_test_trans=${src_test}.m${i}
    tgt_test_trans=${tgt_test}.m${i}

    python ref/inference.py \
           --model_dir ${model_dir} \
           --src ${src_test} \
           --tgt ${tgt_test} \
           --src_vocab ${src_vocab} \
           --tgt_vocab ${tgt_vocab} \
           --direction 1 \
           > ${src_test_trans}
    python ref/inference.py \
           --model_dir ${model_dir} \
           --src ${src_test} \
           --tgt ${tgt_test} \
           --src_vocab ${src_vocab} \
           --tgt_vocab ${tgt_vocab} \
           --direction 2 \
           > ${tgt_test_trans}

    score_test

    # Translate training data.
    src_trans=${src}.m${i}
    tgt_trans=${tgt}.m${i}

    python ref/inference.py \
           --model_dir ${model_dir} \
           --src ${src} \
           --tgt ${tgt} \
           --src_vocab ${src_vocab} \
           --tgt_vocab ${tgt_vocab} \
           --direction 1 \
           > ${src_trans}
    python ref/inference.py \
           --model_dir ${model_dir} \
           --src ${src} \
           --tgt ${tgt} \
           --src_vocab ${src_vocab} \
           --tgt_vocab ${tgt_vocab} \
           --direction 2 \
           > ${tgt_trans}
done
