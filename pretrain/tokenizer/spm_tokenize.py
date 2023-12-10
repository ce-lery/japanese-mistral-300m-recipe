import sentencepiece as spm
from transformers import PreTrainedTokenizer,PreTrainedTokenizerFast,T5Tokenizer
from tokenizers.implementations import SentencePieceUnigramTokenizer

MODEL_PREFIX = "spm-wiki-cc100-for-spm-bytefallback"
OUTPUT_MODEL_DIR = "spm_tokenizer_neologdn_bytefallback_nofast"

spm.SentencePieceTrainer.train(
    input="../dataset/merge_dataset_for_tokenizer.txt",  # コーパスファイル
    model_type="unigram",  # デフォルト
    model_prefix=MODEL_PREFIX,  # 出力されるモデルのファイル名に使われる
    add_dummy_prefix=False,# rinna-3.6bに習って、文章の先頭にスペースが追加されないように
    byte_fallback=True,# rinna-3.6bに習って、未知語をutf-8バイトに分解するために
    # remove_extra_whitespace=False, # rinna-3.6bにならって
    vocab_size=50000,  # vocab number
    character_coverage=0.9995,
    unk_piece="[UNK]",
    pad_piece="[PAD]",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    # train_extremely_large_corpus=True
    input_sentence_size=12000000 # refer:https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part7.html
)

sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PREFIX+".model")

def tokenize(raw_text):
    tokenized=sp.encode_as_pieces(raw_text)
    return tokenized

# encode: text => is
print(sp.encode_as_pieces("これは、テストです。"))
print(sp.encode_as_ids("これは、テストです。"))

# decode: id => text
print(sp.decode_pieces(['▁', 'これは', '、', 'テスト', 'です', '。']))
print(sp.decode_ids([381, 260, 1662, 279, 261]))

# check vocab size
print(sp.get_piece_size())

# Transformer API
tokenizer = T5Tokenizer(
    vocab_file=MODEL_PREFIX+".model",
    unk_token = '[UNK]',
    bos_token = '<s>',
    eos_token = '</s>',
    pad_token = '[PAD]',
    extra_ids=0,
    model_max_length=50000,
)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR) 


