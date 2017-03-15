import gensim


if __name__ == "__main__":
    use_model = gensim.models.Word2Vec.load('govperdue.vecm')
    """:type : gensim.models.Word2Vec"""
    use_model.wv.init_sims()
    print(use_model.wv.similar_by_word('Perdue said.'))