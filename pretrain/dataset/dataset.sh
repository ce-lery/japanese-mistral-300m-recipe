#!/bin/bash

# activate venv
if [ ! -d "../.env_py310" ];then
    echo "Please run ../setup.sh first."
    exit 1
fi
source ../.env_py310/bin/activate

cd dataset

### japanese CC-100 ###
# 0.069TB
echo "# download japanese CC-100"
# wget http://data.statmt.org/cc-100/ja.txt.xz
# unxz -v ja.txt.xz
# # 不要な空行を削除
# sed -i '/^$/d' ja.txt
# head -n15 ja.txt

### japanese wikipedia ###
# 0.004TB
echo "# download japanese wikipedia"
# wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
# bunzip2 jawiki-latest-pages-articles.xml.bz2
# xmlからテキストを抽出
python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml
find text/ | grep wiki | awk '{system("cat "$0" >> wiki.txt")}'
rm -rf text/
head -n100 wiki.txt
#delete doc tag & empty row  
sed -i 's/^<[^>]*>$//g' wiki.txt #<doc>
sed -i '/^$/d' wiki.txt #empty
sed -i '/^ $/d' wiki.txt #space only
head -n100 wiki.txt
# check the column number and word number
wc -ml wiki.txt

# # head wiki.txt -n 10000 > wiki_10000.txt
python wiki_preprocess.py

### dataset creaning ###
# https://www.gnu.org/software/parallel/parallel_tutorial.html
LINES=`wc -l ja.txt | awk '{print $1}'`
cat ja.txt | python normalize.py $LINES > cc100_mrph_for_spm.txt

LINES=`wc -l wiki_preprocess.txt | awk '{print $1}'`
cat wiki_preprocess.txt | python normalize.py $LINES  > wiki_mrph_for_spm.txt

# merge dataset for creating tokenizer
cp wiki_mrph_for_spm.txt merge_dataset_for_tokenizer.txt
cat cc100_mrph_for_spm.txt >> merge_dataset_for_tokenizer.txt

### train test split ###
# cc100_mrph_for_spm.txtをtrain:test = 95:5で分割
echo "# train test split cc100_mrph_for_spm.txt"
LINES=`wc -l cc100_mrph_for_spm.txt | awk '{print $1}'`
echo $LINES

TRAIN_DATA_LINES=$(($LINES*95/100))
TEST_DATA_LINES=$(($LINES-$TRAIN_DATA_LINES))
echo $TRAIN_DATA_LINES
echo $TEST_DATA_LINES

head -n $TRAIN_DATA_LINES cc100_mrph_for_spm.txt > train.txt
tail -n $TEST_DATA_LINES cc100_mrph_for_spm.txt > test.txt

# wiki_mrph_for_spm.txtをtrain:test = 95:5で分割
echo "# train test split wiki_mrph_for_spm.txt"
LINES=`wc -l wiki_mrph_for_spm.txt | awk '{print $1}'`
echo $LINES

TRAIN_DATA_LINES=$(($LINES*95/100))
TEST_DATA_LINES=$(($LINES-$TRAIN_DATA_LINES))
echo $TRAIN_DATA_LINES
echo $TEST_DATA_LINES

head -n $TRAIN_DATA_LINES wiki_mrph_for_spm.txt > wiki_train.txt
tail -n $TEST_DATA_LINES wiki_mrph_for_spm.txt > wiki_test.txt

# merge
cat wiki_train.txt >> train.txt
cat wiki_test.txt >> test.txt
echo "train dataset size:"
wc -l train.txt 
echo "test dataset size:"
wc -l test.txt 

# delete intermediate files
rm wiki_train.txt
rm wiki_test.txt