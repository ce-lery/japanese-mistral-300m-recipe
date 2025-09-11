#! /bin/bash

# Please run the command "cd examples/pretrain_mistral_2b/ ; bash dataset.sh true"

readonly DATA_DIR="./results/dataset/"

# set log
mkdir -p ./results/log/$(basename "$0" .sh)
log=./results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
exec &> >(tee -a $log)
set -x

mkdir -p $DATA_DIR


### download wikipedia, wikibooks, and cc100 corpus ###
# ------------------------------------------
#   Initialize
# ------------------------------------------
mkdir -p ../../third_party/corpus-cleaner/results/dataset/original/

# ------------------------------------------
#   Download wikipedia corpus
# ------------------------------------------
echo ">>> Download Wikipedia."
# wget https://dumps.wikimedia.org/other/cirrussearch/20241007/jawiki-20241007-cirrussearch-content.json.gz -P $DATA_DIR
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/ce-lery/dataset_archive.git $DATA_DIR/dataset_archive
cd $DATA_DIR/dataset_archive
git lfs pull 
mv ./jawiki-20241007-cirrussearch-content.json.gz ../ 
mv ./jawikibooks-20241007-cirrussearch-content.json.gz ../ 
mv ./jawikiversity-20241007-cirrussearch-content.json.gz ../ 
cd -
rm -r $DATA_DIR/dataset_archive

mkdir -p $DATA_DIR/wikipedia
uv run python ../../third_party/bert-japanese/make_corpus_wiki.py \
--input_file $DATA_DIR/jawiki-20241007-cirrussearch-content.json.gz \
--output_file $DATA_DIR/wiki-tohoku-corpus.txt.gz \
--min_sentence_length 10 \
--max_sentence_length 200 \
--mecab_option '-r /usr/local/etc/mecabrc -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd' #echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
uv run python ../../third_party/bert-japanese/merge_split_corpora.py \
--input_files $DATA_DIR/wiki-tohoku-corpus.txt.gz \
--output_dir $DATA_DIR/wikipedia \
--num_files 8
uv run python ../../source/dataset/concat_sentence.py \
--folder_path $DATA_DIR/wikipedia/\*.txt \
--output_file $DATA_DIR/wikipedia/wiki.txt
mv $DATA_DIR/wikipedia/wiki.txt ../../third_party/corpus-cleaner/results/dataset/original/

# ------------------------------------------
#   Download wikibooks corpus
# ------------------------------------------
# download 
echo ">>> Download WikiBooks."
#wget https://dumps.wikimedia.org/other/cirrussearch/20241007/jawikibooks-20241007-cirrussearch-content.json.gz -P $DATA_DIR
mkdir -p $DATA_DIR/wikibooks
uv run python ../../third_party/bert-japanese/make_corpus_wiki.py \
--input_file $DATA_DIR/jawikibooks-20241007-cirrussearch-content.json.gz \
--output_file $DATA_DIR/wikibooks-tohoku-corpus.txt.gz \
--min_sentence_length 10 \
--max_sentence_length 1024 \
--mecab_option '-r /usr/local/etc/mecabrc -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd' #echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
uv run python ../../third_party/bert-japanese/merge_split_corpora.py \
--input_files $DATA_DIR/wikibooks-tohoku-corpus.txt.gz \
--output_dir $DATA_DIR/wikibooks \
--num_files 8
uv run python ../../source/dataset/concat_sentence.py \
--folder_path $DATA_DIR/wikibooks/\*.txt \
--output_file $DATA_DIR/wikibooks/wikibooks.txt
mv $DATA_DIR/wikibooks/wikibooks.txt ../../third_party/corpus-cleaner/results/dataset/original/

# ------------------------------------------
#   Download wikibooks corpus
# ------------------------------------------
echo ">>> Download Wikiversity."
#wget https://dumps.wikimedia.org/other/cirrussearch/20241007/jawikiversity-20241007-cirrussearch-content.json.gz -P $DATA_DIR
mkdir -p $DATA_DIR/wikiversity
uv run python ../../third_party/bert-japanese/make_corpus_wiki.py \
--input_file $DATA_DIR/jawikiversity-20241007-cirrussearch-content.json.gz \
--output_file $DATA_DIR/wikiversity-tohoku-corpus.txt.gz \
--min_sentence_length 10 \
--max_sentence_length 1024 \
--mecab_option '-r /usr/local/etc/mecabrc -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd' #echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
uv run python ../../third_party/bert-japanese/merge_split_corpora.py \
--input_files $DATA_DIR/wikiversity-tohoku-corpus.txt.gz \
--output_dir $DATA_DIR/wikiversity \
--num_files 8
uv run python ../../source/dataset/concat_sentence.py \
--folder_path $DATA_DIR/wikiversity/\*.txt \
--output_file $DATA_DIR/wikiversity/wikiversity.txt

mv $DATA_DIR/wikiversity/wikiversity.txt ../../third_party/corpus-cleaner/results/dataset/original/

# ------------------------------------------
#   download and clean cc100 corpus
# ------------------------------------------
# download cc100
echo ">>> Download japanese CC-100."
wget http://data.statmt.org/cc-100/ja.txt.xz -P $DATA_DIR
mkdir -p $DATA_DIR/cc100/
uv run python ../../third_party/bert-japanese/merge_split_corpora.py \
--input_files $DATA_DIR/ja.txt.xz \
--output_dir $DATA_DIR/cc100/ \
--num_files 64

uv run python ../../source/dataset/concat_sentence.py \
--folder_path $DATA_DIR/cc100/\*.txt \
--output_file $DATA_DIR/cc100/cc100.txt
# rm "./results/dataset/cc100/corpus_${i}.txt"
mv $DATA_DIR/cc100/cc100.txt ../../third_party/corpus-cleaner/results/dataset/original/

# ------------------------------------------
#   download and split oscar2019 corpus
# ------------------------------------------
# reference
# https://huggingface.co/datasets/oscar
# https://huggingface.co/datasets/oscar/blob/main/oscar.py

# download dataset
mkdir -p results/dataset/oscar/original/

for i in $(seq 1 120)
do
    # refer:https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/unshuffled/original/ja/ja_sha256.txt
    wget "https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/unshuffled/original/ja/ja_part_${i}.txt.gz" -P "./results/dataset/oscar/"
    gunzip "./results/dataset/oscar/ja_part_${i}.txt.gz"
    # rm "./results/original/oscar/ja_part_${i}.txt.gz"
done

touch ./results/dataset/oscar/original/oscar2109_ja.txt
merge files of dataset
for i in $(seq 1 120)
do
    cat "./results/dataset/oscar/ja_part_${i}.txt" >> ./results/dataset/oscar/original/oscar2109_ja.txt
    rm  "./results/dataset/oscar/ja_part_${i}.txt"
done

# rm -r "./results/original/oscar"

train test split
echo "# train test split oscar2109_ja.txt"
LINES=`wc -l ./results/dataset/oscar/original/oscar2109_ja.txt | awk '{print $1}'`
echo $LINES

TRAIN_DATA_LINES=$(($LINES*90/100))
REMAIN_DATA_LINES=$(($LINES-$TRAIN_DATA_LINES))
# TEST_DATA_LINES=$((($LINES-$TRAIN_DATA_LINES)/2))
# VALID_DATA_LINES=$(($LINES-$TRAIN_DATA_LINES) - $TEST_DATA_LINES)
echo $TRAIN_DATA_LINES
echo $REMAIN_DATA_LINES
# echo $TEST_DATA_LINES

head -n $TRAIN_DATA_LINES ./results/dataset/oscar/original/oscar2109_ja.txt > ./results/dataset/oscar/original/oscar2109_ja_train.txt
tail -n $REMAIN_DATA_LINES ./results/dataset/oscar/original/oscar2109_ja.txt > ./results/dataset/oscar/original/oscar2109_ja_remain.txt

TEST_DATA_LINES=$(($REMAIN_DATA_LINES/2))
VALID_DATA_LINES=$(($REMAIN_DATA_LINES-$TEST_DATA_LINES))
head -n $TEST_DATA_LINES ./results/dataset/oscar/original/oscar2109_ja_remain.txt > ./results/dataset/oscar/original/oscar2109_ja_test.txt
tail -n $VALID_DATA_LINES ./results/dataset/oscar/original/oscar2109_ja_remain.txt > ./results/dataset/oscar/original/oscar2109_ja_valid.txt

rm ./results/dataset/oscar/original/oscar2109_ja_remain.txt 
rm ./results/dataset/oscar/original/oscar2109_ja.txt 

mv $DATA_DIR/oscar/original/oscar2109_ja_train.txt ../../third_party/corpus-cleaner/results/dataset/original/


# ------------------------------------------
#   corpus_cleaner
# ------------------------------------------
# setup corpus_cleaner
cd ../../third_party/corpus-cleaner/
git checkout v0.1.2
bash scripts/setup.sh
bash scripts/build.sh
cd -

# execute cleaning corpus
cd ../../third_party/corpus-cleaner/corpus_cleaner/build
ICUPATH=$PWD/../../scripts/icu/usr/local
echo $ICUPATH
export C_INCLUDE_PATH=$ICUPATH/include
export CPLUS_INCLUDE_PATH=$ICUPATH/include
export LIBRARY_PATH=$ICUPATH/lib
export LD_LIBRARY_PATH=$ICUPATH/lib

./corpus_cleaner
cd -

# ------------------------------------------
#   create corpus
# ------------------------------------------
### merge jsonl files ###
cp ../../third_party/corpus-cleaner/results/dataset/cleaned/wiki.jsonl ./results/dataset/train.jsonl
cat ../../third_party/corpus-cleaner/results/dataset/cleaned/wikibooks.jsonl >>./results/dataset/train.jsonl
cat ../../third_party/corpus-cleaner/results/dataset/cleaned/wikiversity.jsonl >>./results/dataset/train.jsonl
cat ../../third_party/corpus-cleaner/results/dataset/cleaned/cc100.jsonl >> ./results/dataset/train.jsonl
cat ../../third_party/corpus-cleaner/results/dataset/cleaned/oscar2109_ja_train.jsonl >> ./results/dataset/train.jsonl

### merge jsonl files for tokenizer ###
mkdir -p ./results/preprocess/
cp ../../third_party/corpus-cleaner/results/dataset/cleaned/wiki.jsonl ./results/tokenizer/tokenizer_corpus.jsonl
cat ../../third_party/corpus-cleaner/results/dataset/cleaned/cc100.jsonl >> ./results/tokenizer/tokenizer_corpus.jsonl

mv ../../third_party/corpus-cleaner/results/dataset/cleaned/oscar2109_ja_valid.jsonl ./results/dataset/
mv ../../third_party/corpus-cleaner/results/dataset/cleaned/oscar2109_ja_test.jsonl ./results/dataset/

# split dataset for pretrain
uv run python split_dataset.py \
    --input_file ./results/dataset/train.jsonl \
    --split_ratio 0.8
    # --input_file ../../source/dataset/wiki.jsonl \

### remove directory ###
# for dataset_type in "oscar" "wiki" "cc100"
# do
# rm -r ./third_party/corpus-cleaner/results/data/$dataset_type/cleaned/ 
# done
