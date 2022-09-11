import os
import argparse
import numpy as np
import pandas as pd
import pickle

from gensim.models import FastText
from fse.models import SIF
from fse import IndexedList
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
            
            if key.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                key = 'chr' + key
            corpus.append(sentence)
            sentence_name.append(key)    

        return corpus, sentence_name

    def extract_sentence_vectors(self, model, sentences,sentence_name): 

        sentences_vector= {}
        for x in range(len(sentences)):
            
            vectors = [model.wv[w] for w in sentences[x]
                   if w in model.wv]

            new_vector = np.zeros(model.vector_size)

            new_vector = (np.array([sum(t) for t in vectors])) / new_vector.size

            sentences_vector[sentence_name[x]] = new_vector
            print(len(new_vector))

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
    
    ft = FastText(sentences, min_count=1,
                             window=7,
                             vector_size=4,
                             seed=777,
                             workers=15)
    model = SIF(ft)
    model.train(IndexedList(sentences))
    sentence_vectors = np.around(model.sv.vectors, 3)
    print(sentence_vectors)
    sentence_dict = {}

    for x in range(len(sentence_name)):
        sentence_dict[sentence_name[x]] = sentence_vectors[x,]
        
    save_path = os.path.split(args.input_file)[0]

 #    print(np.zeros(model.vector_size))

 # #   model.wv.save_word2vec_format(save_path + '/dna2vec_embed.emb')
 #    sentences_vector = dna2vec.extract_sentence_vectors(model, sentences, sentence_name)
   # print(sentences_vector)
    with open(save_path + '/dna2vec_sequence_embed.pkl', 'wb') as f:
        pickle.dump(sentence_dict, f)


if __name__ == "__main__":
    main()
 

#python dna2vec.py -i ../results/peak_regions.fa -k 6