#!/bin/bash

echo "Start mkdir "
mkdir -p -m 777 /eval
mkdir -p -m 777 /transformers_cache

echo "Start copy files"
pushd text-to-sql
cp -r seq2seq /app/
cp -r configs /app/
popd

echo "Start copy data"
cp -r data /
cp -r database /

echo "pwd and ls"
pwd
ls

echo "Start download stanza"
pip install stanza

pushd /app
python -c "import transformers;print(transformers.__version__)"
python3 seq2seq/stanza_downloader.py

echo "pwd and ls"
pwd
ls


echo "Start running codalab_seq2seq"
CUDA_VISIBLE_DEVICES="-1" python3 /app/seq2seq/eval_run_seq2seq.py /app/configs/test_t5_3b_2112_sparc.json
echo "Finished running codalab_seq2seq"

popd
echo "pwd and ls"
pwd
ls

echo "pwd and ls"
cp /eval/predicted_sql.txt ./predicted_sql.txt
rm -rf ./stanza_resources
