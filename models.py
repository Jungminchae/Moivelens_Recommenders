import os
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from implicit.bpr import BayesianPersonalizedRanking
import random 
from IPython.display import display
from dataloader import Movielens_Dataloaders

class Weighted_popularity_recommender:
    def top_20_recommendation(self, df):
        # top20 추천
        df = df.sort_values(by='weighted_rating', ascending=False).reset_index(drop=True)
        top_20 = df.iloc[:20]
        display(top_20[['title','release_year','weighted_rating']])
    def top_20_random_5_recommendation(self, df):
        # top20위 중 5개 영화 랜덤 추천 
        df = df.sort_values(by='weighted_rating', ascending=False).reset_index(drop=True)
        top_20 = df.iloc[:20]
        random_index = random.sample(range(0,19),5)
        display(top_20[['title','release_year','weighted_rating']].loc[random_index])

class Apriori_recommender(Movielens_Dataloaders):
    def __init__(self):
        super().__init__()
        self.movie_df = pd.read_pickle(self.movie_data_path)
    def train_apriori(self, df, sample_size=1):
        freq_sets_df = apriori(df.sample(frac=sample_size),
                       min_support=0.01, # 1% 이상 포함
                       max_len=2, # 빈발집합의 최대 크기
                       use_colnames=True,
                       verbose=1)
        return freq_sets_df

    def id2title(self, df):
        # movie_id 를 movie_title로 바꿔주기
        id_to_title = dict(zip(self.movie_df["movie_id"], self.movie_df.title))
        df.antecedents = (
        df.antecedents
            .apply(lambda x : list(x)[0]) 
            .apply(lambda x : id_to_title[x]))
        return df 

    def pooling_relationship(self, df):
        # 연관관계 추출하기
        item_rules = association_rules(df, 
                               metric='support',
                               min_threshold=0.01)
        item_rules = item_rules[item_rules.confidence > 0.1]
        item_rules = item_rules.sort_values('lift', ascending=False)
        return item_rules
    
    def show_5_movies(self, df, movie_name):
        # 연관관계 높은 영화 5개 추천
        display(df[df.antecedents == movie_name].iloc[:5])


class Maxtrix_Factorization(Movielens_Dataloaders):
    def __init_(self):
        super().__init__()
        self.movie_df = pd.read_pickle(self.movie_data_path)
    def BPR_train(self, inputs, rating_df):
        model = BayesianPersonalizedRanking(factors=60)
        model.fit(inputs)

        # user_embeddings = model.user_factors
        movie_embeddings = model.item_factors

        # id를 영화 이름으로 변경
        id_title_dict = { k:v for k,v in self.movie_df['title'].items()}
        title = [id_title_dict[movie_id] for movie_id in rating_df['movie_id'].cat.categories]  

        # movie embedding
        movie_embedding_df = pd.DataFrame(movie_embeddings, index=title)
        # user_names = [user_id for user_id in rating_df['user_id'].cat.categories]
        # user_embedding_df = pd.DataFrame(user_embeddings, index=user_names)

        return movie_embedding_df # , user_embedding_df

    def BPR_recommendation(self, movie_embedding_df, selected_movie):
        target = movie_embedding_df.loc[selected_movie]
        # dot product
        # selected_movie와 유사한 영화 상위 10개 추천
        reco  = movie_embedding_df.dot(target).sort_values(ascending=False).iloc[:10]
        return display(reco)



class Simple_Collaborative_Filtering:
    pass

class Hybrid_recommender:
    pass
