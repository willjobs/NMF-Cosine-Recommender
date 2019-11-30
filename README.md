# NMF-Cosine-Recommender
Movie recommendation algorithm combing NMF (non-negative matrix factorization) with cosine similarity.

This combines the collaborative filtering approach of NMF, with the content filtering approach of cosine similarity. 

The main dependencies are scikitlearn (for cosine similarity) and surprise (for NMF). Scikitlearn has a major flaw in its NMF implementation which is that missing values aren't handled very well. For sparse matrices (like a set of user reviews), if you initialize this missing values to 0, the imputed values from NMF will be close to 0. The surprise library correctly fits to the ratings that exist.
