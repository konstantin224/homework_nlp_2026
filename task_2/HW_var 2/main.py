
import sys
import argparse

import numpy as np

from config import *
from gensim.models import Word2Vec
from preprocess import preprocessing_text



def open_file(path_file):
    
    with open(path_file, "r", encoding="utf-8") as f:
        lines_list = f.readlines()

    return lines_list

def load_word2vec_model(path_model):

    model_word2vec = Word2Vec.load(path_model)

    return model_word2vec

def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def word_to_vec(word, model):
    
    if model is None:
        print("Модель не загружена.", file=sys.stderr)
        return None

    if not hasattr(model, 'wv'):
        print("Переданный объект не является корректной моделью Word2Vec.", file=sys.stderr)
        return None

    try:
        return model.wv[word]
    
    except KeyError:
        return None
    
    except Exception as e:
        print(f"Ошибка при получении вектора для '{word}': {type(e).__name__}: {e}", file=sys.stderr)
        return None


def semantic_search(query, query_vec, sentence, 
                    model, treshold: float = 0.8):
    
    if query_vec is None:
        return in_word_search(query, sentence)
    
    sentence_vec = list(map(lambda x: word_to_vec(x, model), sentence))

    for word_vec in sentence_vec:
        
        if word_vec is None:
            continue

        sim = cosine_sim(query_vec, word_vec)

        if sim > treshold:
            return "is_similarity"

    return in_word_search(query, sentence)

def in_word_search(query, sentence):

    if query in sentence:
        return "is_similarity"
    
    return "is_not_similarity"

def search_sim_sentence(query, query_vec, list_sentence, model):
    
    final_sentence = []
    final_answer = "not_finded_sentence"

    for sentence in list_sentence:
        
        responce_answer = semantic_search(query, query_vec, sentence, model)
        
        if responce_answer == "is_similarity":
            final_sentence.append(sentence)
            final_answer = "finded_sentence"

    return {
        "answer": final_answer,
        "sentence": final_sentence
    }
        
def main():


    parser = argparse.ArgumentParser(description="Семантический grep с поддержкой синонимов")
    parser.add_argument("file", help="Путь к текстовому файлу")
    parser.add_argument("query", help="Слово для поиска")
    args = parser.parse_args()

    
    files = open_file(args.file)
    model_word2vec = load_word2vec_model(PATH_WORD2VEC_MODEL)

    query_vec = word_to_vec(args.query, model_word2vec)

    responce = search_sim_sentence(args.query, query_vec, files, model_word2vec)

    if responce['answer'] == "finded_sentence":
        text = f"Алгоритм закончил работу и нашёл следующие предложения:\n"
        print(text)
        for sentence in responce['sentence']:
            text = sentence + "\n"
            print(text)
    else:
        text = "Алгоритм закончил работу и не нашёл предложений. Пожалуйста, переформулируйте запрос."

if __name__ == "__main__":
    main()