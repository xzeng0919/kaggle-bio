import os
import argparse
import numpy as np
import pandas as pd
import pickle

from gensim.models import Word2Vec
from Bio import SeqIO


class Dna2vec:

    def parser_fasta_file(self, file_path):

        fasta_seqs = SeqIO.parse(open(file_path), 'fasta')
        fasta_seqs_dict = SeqIO.to_dict(fasta_seqs)

        return fasta_seqs_dict

    def get_kmer_sequence(self, sequence, k):

        sentence = []

        for x in range(len(sequence) - k + 1):
            word = sequence[x:x+k]
            sentence.append(word)

        return sentence

    def get_ker_sequence_dict(self, sequence_dict, k):

        corpus = []
        sentence_name = []
        for key,value in sequence_dict.items():
            sequence = str(sequence_dict[key].seq)
            sentence = self.get_kmer_sequence(sequence, k)
            
            corpus.append(sentence)
            sentence_name.append(key)    

        return corpus, sentence_name

    def extract_sentence_vectors(self, model, sentences,sentence_name): 

        sentences_vector= {}
        for x in range(len(sentences)):
            sentences_vector[sentence_name[x]] = model.wv[sentences[x]]

        return sentences_vector    


def main():

    parser = argparse.ArgumentParser(
        description="DNA sequences embedding by Word2Vec")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        action="store",
        type=str,
        required=True,
        help="Input fasta filename.")

    parser.add_argument(
        "-k",
        "--kmer",
        dest="k",
        action="store",
        type=int,
        required=True,
        help="k-mer for DNA sequences")

    args = parser.parse_args()

    dna2vec = Dna2vec()
    sequence_dict = dna2vec.parser_fasta_file(args.input_file)
    sentences,sentence_name = dna2vec.get_ker_sequence_dict(sequence_dict,args.k)
    
    model = Word2Vec(sentences = sentences,
                        min_count= 1,
                        vector_size=8, 
                        sg=0, 
                        hs=1,
                        workers=15, 
                        window=7,
                        seed=777,
                        epochs=5)
    save_path = os.path.split(args.input_file)[0]
    model.wv.save_word2vec_format(save_path + '/dna2vec_embed.emb')
    sentences_vector = dna2vec.extract_sentence_vectors(model, sentences, sentence_name)

    with open(save_path + '/dna2vec_sequence_embed.pkl', 'wb') as f:
        pickle.dumps(sentences_vector, f)


if __name__ == "__main__":
    main()
 