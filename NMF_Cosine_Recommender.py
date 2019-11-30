import numpy as np
import pandas as pd
import random
from surprise import Reader, Dataset
from surprise import NMF
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NMF_Cosine_Recommender:
    """[summary]
       @author Will Jobs
    """
    def __init__(self, df_users, df_movies, df_ratings, df_movie_lens_tags, biased=False):
        """[summary]

        Args:
            df_users ([type]): [description]
            df_movies ([type]): [description]
            df_ratings ([type]): [description]
            df_movie_lens_tags ([type]): [description]
            biased
        """
        self.users = df_users
        self.movies = df_movies
        self.ratings = df_ratings
        self.ml_tags = df_movie_lens_tags
        self.biased = biased
        self.trained_nmf = False
        self.preprocessed = False
        self.trained_cosine = False
        self.cv_score = None
        self.cv_fit_time = None
        self.movies_merged = pd.DataFrame()
        self.nmf_predictions = pd.DataFrame()
        self.tfidf_matrix = None
        self.algo = None
        self.W = None
        self.H = None

    def preprocess_tags(self, verbose=True):
        """[summary]

        Args:
            verbose (bool, optional): [description]. Defaults to True.
            seed ([type], optional): [description]. Defaults to None.
        """
        if self.preprocessed:  # only do this once
            return

        if verbose:
            print('Preprocessing tags and movie information...', end='')

        self.ml_tags.rename(columns={'userId':'userID','movieId':'movieID'}, inplace=True)
        self.ml_tags = self.ml_tags.astype({'tag':str})

        tmp_tags = self.ml_tags.copy()
        tmp_movies = self.movies.copy()

        # replace punctuation in tags (a space), movie name (a space), and genres (no space). These will eventually be folded into the tags list
        # doing it this way to avoid altering the original tags during presentation later
        tmp_tags['new_tag'] = tmp_tags.tag.str.replace(r'[^\w\s]',' ')
        tmp_movies['new_name'] = tmp_movies.name.str.replace(r'[^\w\s]',' ')
        tmp_movies['new_genre1'] = tmp_movies.genre1.str.replace(r'[^\w\s]','')
        tmp_movies['new_genre2'] = tmp_movies.genre2.str.replace(r'[^\w\s]','')
        tmp_movies['new_genre3'] = tmp_movies.genre3.str.replace(r'[^\w\s]','')

        # aggregate all users' tags up per movie
        tags_nostrip = tmp_tags.groupby('movieID').tag.apply(' '.join).reset_index()
        tags_nostrip.rename(columns={'tag':'tags'}, inplace=True)
        tags_strip = tmp_tags.groupby('movieID').new_tag.apply(' '.join).reset_index()
        tags_strip = tags_nostrip.merge(tags_strip, on='movieID')

        # merge name, genres, and tags together
        self.movies_merged = tmp_movies.merge(tags_strip, on='movieID', how='left')
        self.movies_merged['tags_strip'] = self.movies_merged.apply(lambda x: '{} {} {} {} {}'.format(x['new_name'], x['new_genre1'], x['new_genre2'] if type(x['new_genre2']) != float else "", x['new_genre3'] if type(x['new_genre3']) != float else "", x['new_tag']), axis=1)
        self.movies_merged.drop(columns=['new_tag','new_name','new_genre1','new_genre2','new_genre3'], inplace=True)
        
        # merge in the combined tags (with punctuation)
        self.movies = self.movies.merge(tags_nostrip, on='movieID', how='left')

        self.preprocessed = True

        if verbose:
            print('Done')

    def train_cosine_similarity(self, seed=None, verbose=True):
        """[summary]

        Args:
            seed ([type], optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to True.

        Raises:
            RuntimeError: [description]
        """
        if not self.preprocessed:
            raise RuntimeError('Cannot train cosine similarity until preprocessing is done (via preprocess_tags)')

        if self.trained_cosine: # only do this once
            return

        if seed is not None:
            random.seed(seed)

        vectorizer = TfidfVectorizer(stop_words='english', min_df=3)

        if verbose:
            print('Cosine similarity training...', end='')

        self.tfidf_matrix = vectorizer.fit_transform(self.movies_merged['tags_strip'])
        self.trained_cosine = True

        if verbose:
            print('Done')

    def run_nmf(self, n_factors=15, run_cross_validation=True, cv_metric='RMSE', seed=None, verbose=True):
        """[summary]

        Args:
            n_factors (int, optional): [description]. Defaults to 15.
            run_cross_validation (bool, optional): [description]. Defaults to True.
            cv_metric (str, optional): [description]. Defaults to 'RMSE'.
            seed ([type], optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to True.
        """

        # ratings get clipped from 1 to 5
        reader = Reader(rating_scale=(1.0,5.0))
        data = Dataset.load_from_df(self.ratings, reader)

        # first, calculate CV on a fraction of the dataset
        if run_cross_validation:
            if verbose:
                print('Running cross-validation...', end='')

            if seed is not None:
                random.seed(seed)

            algo = NMF(n_factors=n_factors, biased=self.biased, random_state=seed)
            cv_results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)
            avg_cv_result = pd.DataFrame.from_dict(cv_results).mean(axis=0)
            self.cv_score = avg_cv_result['test_' + cv_metric.lower()]
            self.cv_fit_time = avg_cv_result['fit_time']

            if verbose:
                print('Done')
                print('Average CV score: {}\nAverage fit time: {} seconds'.format(round(self.cv_score, 4), round(self.cv_fit_time, 4)))

        if seed is not None:
            random.seed(seed)

        # ratings must have 3 cols: users, items, ratings (in that order)
        train_set = data.build_full_trainset()

        self.algo = NMF(n_factors=n_factors, biased=self.biased, random_state=seed)

        if verbose:
            print('NMF Fitting...', end='')

        self.algo.fit(train_set)

        self.W = self.algo.pu
        self.H = np.transpose(self.algo.qi)

        # get predictions for *every* user/movie combo. These will be also compared to the actual ratings
        if verbose:
            print('Done')
            print('Generating all user-movie pairs for predictions...', end='')

        all_pairs = [(x, y, 0) for x in self.users.userID for y in self.movies.movieID]

        # getting predictions for ALL user/movie combos
        # took 40 seconds on 3.4 million rows
        if verbose:
            print('Done')
            print('Calculating predictions on all user-movie pairs...', end='')

        all_preds = self.algo.test(all_pairs)
        all_preds = pd.DataFrame([{'userID':y.uid, 'movieID':y.iid, 'nmf_prediction':y.est} for y in all_preds])

        self.nmf_predictions = all_preds.merge(self.ratings, on=['userID','movieID'], how='left')
        self.nmf_predictions = self.nmf_predictions[['userID','movieID','rating','nmf_prediction']]
        self.trained_nmf = True

        if verbose:
            print('Done')

    def train(self, n_factors=15, run_cross_validation=True, seed=None, verbose=True):
        """[summary]

        Args:
            n_factors (int, optional): [description]. Defaults to 15.
            run_cross_validation (bool, optional): [description]. Defaults to True.
            seed ([type], optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to True.
        """
        self.preprocess_tags(verbose=verbose)
        self.train_cosine_similarity(seed=seed, verbose=verbose)
        self.run_nmf(n_factors=n_factors, run_cross_validation=run_cross_validation, seed=seed, verbose=verbose)

    def get_similar_movies(self, movieID, number_of_movies=None, verbose=True):
        """[summary]

        Args:
            movieID ([type]): [description]
            verbose (bool, optional): [description]. Defaults to True.

        Raises:
            RuntimeError: [description]

        Returns:
            [type]: [description]
        """
        if not (self.preprocessed and self.trained_cosine):
            raise RuntimeError('Cannot make recommendations without training NMF, preprocessing, and training cosine first.')

        # get the index of the movie
        idx = np.where(self.movies_merged.movieID==movieID)[0][0]

        if verbose:
            print('Getting similar movies to ' + self.movies_merged.iloc[idx]['name'] + '...', end='')

        y = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)
        idx_scores = pd.DataFrame([(idx, score) for (idx, score) in enumerate(list(y[0])) if score>0], columns=['idx','similarity'])

        result = pd.concat([self.movies_merged.iloc[idx_scores.idx].reset_index(), idx_scores], axis=1).sort_values(by='similarity', ascending=False)

        # get rid of transformed columns from movies_merged (except tag), and get the *original* name and genres with punctuation
        result.drop(columns=[x for x in [*self.movies_merged.columns, 'index','idx'] if x != 'movieID'], inplace=True)
        result = result.merge(self.movies, on='movieID', how='left')
        result = result[['movieID','name','year','genre1','genre2','genre3','tags','similarity']]

        if verbose:
            print('Done')

        # don't include the movie we're finding similarities for
        if number_of_movies is not None:
            return result[1:].head(number_of_movies)
        else:
            return result[1:]

    def get_recommendations(self, userID, number_of_recs=5, seed=None, show_user_likes=True, verbose=True):
        """[summary]
        Algorithm:
        1. Get 20 of the users' top ratings. Start with 5s, if > 20 exist, sample 20 randomly.
        2. If fewer than 20 5s exist, sample 4s until get to 20 (or use up all 4s).
            - If there are no 5s or 4s, ignore the user's ratings, and just
              return the <number_of_recs> top predicted ratings for this user. Done.
        3. For each movie in the top list, calculate cosine similarity, and get the 10 most-similar
           movies which the user has NOT seen.
        4. Combine the 20 most-similar lists of 10 movies into a single list.
        5. Remove duplicates from this list, choosing the highest-similarity achieved
        6. For each movie, look up the predicted rating for this user.
        7. Multiply each movie's similarity times the predicted rating.
        8. Return the top <number_of_recs> predicted movies (or all if not enough). Done.

        Args:
            userID ([type]): [description]
            number_of_recs (int, optional): [description]. Defaults to 5.
            seed ([type], optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to True.

        Returns:
            pandas DataFrame: expected ratings. Columns: movieID, name, genres, weighted_rating
        """
        MAX_CONSIDERED_RATINGS = 20
        CONSIDER_N_SIMILAR = 10

        def combine_genres(df):
            # combine genres into a single column. Note that NaNs parse as float during apply
            df['genres'] = df.apply(lambda row: (row['genre1'] if not type(row['genre1'])==float else "") + \
                                                ("/" + row['genre2'] if not type(row['genre2'])==float else "") + \
                                                ("/" + row['genre3'] if not type(row['genre3'])==float else ""), axis=1)
            df.drop(columns=['genre1','genre2','genre3'], inplace=True)

        def get_subset_ratings():
            if verbose:
                print("Getting user's highest rated movies to start from...", end='')

            all_5s = self.ratings[(self.ratings.userID==userID) & (self.ratings.rating==5)]

            if len(all_5s) >= MAX_CONSIDERED_RATINGS:
                subset_ratings = all_5s.sample(MAX_CONSIDERED_RATINGS, random_state=seed)
            else:
                # use all 5s, and add in 4s until we have <MAX_CONSIDERED_RATINGS>
                subset_ratings = all_5s.copy()
                all_4s = self.ratings[(self.ratings.userID==userID) & (self.ratings.rating==4)]
                count_needed = MAX_CONSIDERED_RATINGS - len(all_5s)
                subset_ratings = pd.concat([subset_ratings, all_4s.sample(min(count_needed, len(all_4s)), random_state=seed)], ignore_index=True)

            subset_ratings = subset_ratings.merge(self.movies[['movieID','name']], on='movieID')

            if verbose:
                print('Done')

            return subset_ratings[['userID','movieID','name','rating']]

        def get_most_similar_movies(subset_ratings):
            if verbose:
                print("Finding similar movies to {} movies the user liked...".format(len(subset_ratings)))

            seen_movies = list(self.ratings[(self.ratings.userID==userID)].movieID)
            similar_movies = pd.DataFrame()

            for movie in subset_ratings.movieID:
                tmp_similar = self.get_similar_movies(movie, verbose=verbose)

                # limit to movies the user hasn't seen, and limit to top <CONSIDER_N_SIMILAR>
                tmp_similar = tmp_similar[~tmp_similar['movieID'].isin(seen_movies)].head(CONSIDER_N_SIMILAR)
                tmp_similar['similar_to'] = subset_ratings[subset_ratings['movieID']==movie].name.values[0]
                similar_movies = pd.concat([similar_movies, tmp_similar], ignore_index=True)

            # now remove duplicates, and get the top similarity for each movie
            similar_movies.sort_values(by='similarity', ascending=False, inplace=True)
            similar_movies.drop_duplicates(subset='movieID', keep='first', inplace=True)

            return similar_movies

        if not (self.trained_nmf and self.preprocessed and self.trained_cosine):
            raise RuntimeError('Cannot make recommendations without training NMF, preprocessing, and training cosine first.')

        if userID not in self.users.userID.values:
            raise ValueError('User {} does not exist in ratings dataset. If this is a new user, create a new user using the average ratings.'.format(userID))

        if seed is not None:
            random.seed(seed)

        review_counts = self.ratings[self.ratings.userID==userID].rating.value_counts()

        if review_counts.get(5,0) + review_counts.get(4,0) == 0:
            # ignore user's ratings, and just get the user's top <number_of_recs> ratings
            if verbose:
                print("User has no ratings >= 4. Ignoring user's ratings, returning top predicted ratings.")

            # get only predicted ratings for ones the user hasn't seen
            subset_ratings = self.nmf_predictions.loc[(self.nmf_predictions.userID == userID) & (self.nmf_predictions.rating.isna())].copy()
            subset_ratings = subset_ratings.merge(self.movies, on='movieID')
            combine_genres(subset_ratings)

            # add in columns that would have been calculated
            subset_ratings['similar_to'] = ""
            subset_ratings['similarity'] = np.nan
            subset_ratings['weighted_rating'] = subset_ratings['nmf_prediction']
            
            # reorder columns
            subset_ratings = subset_ratings[['movieID','name','year','genres','tags','similar_to','similarity','nmf_prediction','weighted_rating']]
            subset_ratings.sort_values(by='nmf_prediction', ascending=False, inplace=True)

            return subset_ratings.head(number_of_recs)

        # get up to <MAX_CONSIDERED_RATINGS> 5s
        subset_ratings = get_subset_ratings()

        if show_user_likes:
            print('\n---------------\nHighest-reviewed movies for userID {}:'.format(userID))
            print(subset_ratings)
            print('\n---------------\n')

        # get the similarity for each movie in subset_ratings
        similar_movies = get_most_similar_movies(subset_ratings)

        # now we have the similarity scores for the movies most like the movies the user rated highest
        # get the predicted ratings, and multiply those by the similarity scores
        if verbose:
            print("Getting user's predicted ratings and calculated expected rating...", end='')

        user_predictions = self.nmf_predictions[self.nmf_predictions['userID']==userID]
        similar_movies = similar_movies.merge(user_predictions, on='movieID', how='inner')
        similar_movies['weighted_rating'] = similar_movies['similarity'] * similar_movies['nmf_prediction']

        if verbose:
            print('Done')
            print("Finalizing output...", end='')

        # combine genres and reorder columns
        combine_genres(similar_movies)
        similar_movies = similar_movies[['movieID','name','year','genres','tags','similar_to','similarity','nmf_prediction','weighted_rating']]
        similar_movies.sort_values(by='weighted_rating', ascending=False, inplace=True)

        if verbose:
            print('Done')

        return similar_movies.head(number_of_recs)
