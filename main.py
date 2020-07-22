from dataloader import Movielens_Dataloaders
from models import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reco', choices=['popularity', 'apriori', 'bpr'], default='popularity')
    parser.add_argument('--genre', type=str)
    parser.add_argument('--threshold',type=int)
    parser.add_argument('--p_random', choices=['random', 'no_random'], default='no_random')
    parser.add_argument('--movie_name',type=str)
    parser.add_argument('--selected_movie', type=str)
    args = parser.parse_args()





    data_load = Movielens_Dataloaders()

    # Recommenders
    if args.reco =='popularity':
        dfp = data_load.data_for_popularity(genre_choice=args.genre)
        model = Weighted_popularity_recommender()
        if args.p_random =='no_random':
            model.top_20_recommendation(dfp)
        else:
            model.top_20_random_5_recommendation(dfp)

    elif args.reco =='apriori':
        dfa = data_load.data_for_apriori()
        model = Apriori_recommender()
        feq = model.train_apriori(dfa)
        feq = model.id2title(feq)
        item_rules = model.pooling_relationship(feq)
        model.show_5_movies(item_rules, movie_name=args.movie_name)

    elif args.reco =='bpr':
        dfb, rating_df = data_load.data_for_bpr(threshold=args.threshold)
        model = Maxtrix_Factorization()
        movie_embedding = model.BPR_train(dfb, rating_df)
        model.BPR_recommendation(movie_embedding, selected_movie=args.selected_movie)

    