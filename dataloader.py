import pandas as pd
import numpy as np
import os 
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix



class Movielens_Dataloaders:
    def __init__(self):
        self.rating_data_path = os.path.join('data','rating_df.pkl')
        self.movie_data_path = os.path.join('data','movie_df.pkl')
        self.genre_data_path = os.path.join('data','genre_df.pkl')

    def data_for_apriori(self):
        # rating_data
        rating_df = pd.read_pickle(self.rating_data_path)
        
        # 4점 이상의 고평점 영화만 사용
        over_4_rating = rating_df[rating_df['rating'] >=4]
        user_movie_basket = over_4_rating.groupby('user_id')['movie_id'].apply(set)

        # basket -> vector
        transaction = TransactionEncoder()
        basket_array = transaction.fit_transform(user_movie_basket)

        basket_df = pd.Dataframe(basket_array, columns = transaction.columns_)

        # 평점 개수 기준 상위 5000개
        top_5000_movie= rating_df.groupby('movie_id')['rating'].count().sort_values(ascending=False).\
            iloc[:5000].index
        top_5000_basket = basket_df[top_5000_movie]
        top_5000_basket = top_5000_basket[top_5000_basket.sum(axis=1) > 0]

        return top_5000_basket

    def data_for_popularity(self, genre_choice=None):
        # movie_data
        movie_df = pd.read_pickle(self.movie_data_path)
        rating_df = pd.read_pickle(self.rating_data_path)
        genre_df = pd.read_pickle(self.genre_data_path)

        # 가중치를 만들기 위한 data handling
        movie_rating_counts = rating_df.groupby('movie_id')['rating'].count()
        movie_rating_mean = rating_df.groupby('movie_id')['rating'].mean()
        concat_df = pd.concat([movie_rating_counts, movie_rating_mean] , axis=1)
        concat_df.columns = ['count', 'mean_rating']
        concat_df = pd.merge(concat_df,genre_df, on='movie_id')
        
        def weighted_rating(df):
            v = df['count']
            R = df['mean_rating']
            m = v.quantile(0.95)
            C = R.mean()
            df['weighted_rating'] =  (v / (v+m) * R) + (m / (m+v) *C)
            return df
        if genre_choice == None:
            # concat한 df에 weighted_rating 추가
            concat_df = concat_df.drop('genre', axis=1).drop_duplicates()
            concat_df = weighted_rating(concat_df)
            # 최종 df
            final_df = pd.merge(movie_df, concat_df, left_on='id', right_on='movie_id').drop('id',axis=1)
        else:
            by_genre_df = concat_df[concat_df['genre'].str.contains(genre_choice)].reset_index(drop=True)
            by_genre_df = weighted_rating(by_genre_df)
            final_df = pd.merge(movie_df, by_genre_df, left_on='id', right_on='movie_id').drop('id',axis=1)
        return final_df
        
    def data_for_bpr(self, threshold=5):
        # movie_data
        rating_df = pd.read_pickle(self.rating_data_path)

        # threshold 보다 많은 사람이 본 영화 
        user_per_movie = (
            rating_df.groupby('movie_id')['user_id']
            .count()
        )
        over_movie_ids = user_per_movie[user_per_movie > threshold].index

        # threshold 보다 영화를 많이 본 사람
        movie_per_user = (
            rating_df.groupby('user_id')['movie_id']
            .count()
        )
        over_user_ids = movie_per_user[movie_per_user > threshold].index

        rating_df = rating_df[
            (rating_df['user_id'].isin(over_user_ids)) 
            & (rating_df['movie_id'].isin(over_movie_ids))
        ]

        # user_id , movie_id 카테고리
        rating_df['user_id'] = pd.Categorical(rating_df['user_id'])
        rating_df['movie_id'] = pd.Categorical(rating_df['movie_id'])

        user_idx = rating_df['user_id'].cat.codes.values
        movie_idx = rating_df['movie_id'].cat.codes.values

        value = rating_df['rating'].values.astype(np.float32)
        row = movie_idx.copy()
        col = user_idx.copy() 

        inputs = csr_matrix((value, (row,col)))

        return inputs, rating_df

