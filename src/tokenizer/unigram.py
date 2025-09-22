import sentencepiece as spm
from transformers import LlamaTokenizer
import argparse
from argparse import Namespace

def train_sentencepiece_tokenizer(corpus_path:str, 
                                  model_prefix:str,
                                  vocab_size:int,
                                  output_dir:str)->None:

    spm.SentencePieceTrainer.train(
        input=corpus_path,  # corpus file
        model_type="unigram",  # default parameter
        model_prefix=model_prefix,  # Used for the file name of the output model
        add_dummy_prefix=False,# Learn from rinna-3.6b and avoid adding spaces at the beginning of sentences.
        byte_fallback=True,# Learning from rinna-3.6b, to decompose unknown words into UTF-8 bytes
        vocab_size=vocab_size,  # vocab number
        character_coverage=0.9995,
        unk_piece="[UNK]",
        pad_piece="[PAD]",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        #train_extremely_large_corpus=True,
        input_sentence_size=1280000 # refer:https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part7.html
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(model_prefix+".model")

    # encode: text => is
    print(sp.encode_as_pieces("これは、テストです。"))
    print(sp.encode_as_ids("これは、テストです。"))

    # decode: id => text
    print(sp.decode_pieces(['▁', 'これは', '、', 'テスト', 'です', '。']))
    print(sp.decode_ids([423, 260, 1866, 277, 261]))

    # Transformer API
    tokenizer = LlamaTokenizer(
        vocab_file=model_prefix+".model",
        unk_token = '[UNK]',
        bos_token = '<s>',
        eos_token = '</s>',
        pad_token = '[PAD]',
        extra_ids=0,
        model_max_length=vocab_size,
        legacy=False
    )
    tokenizer.save_pretrained(output_dir) 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_prefix",help="",default="")
    parser.add_argument("--output_dir",help="",default="")
    parser.add_argument("--corpus_path",help="",default="")
    parser.add_argument("--vocab_size",help="",type=int,default="32000") # 31743+257(fallback)=3200

    args = parser.parse_args()
    train_sentencepiece_tokenizer(corpus_path=args.corpus_path,
                                  model_prefix=args.model_prefix,
                                  vocab_size=args.vocab_size,
                                  output_dir=args.output_dir)
