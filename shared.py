import pandas as pd
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def load_data():
    train_df = pd.read_csv('train_reviews_santa_barbara.csv')
    test_df = pd.read_csv('test_reviews_santa_barbara.csv')
    restaurants_df = pd.read_csv('restaurants_santa_barbara.csv')
    return train_df, test_df, restaurants_df


def evaluate_model(predictions, test_df, k=10):
    test_dict = test_df.groupby('user_id').apply(
        lambda x: dict(zip(x['business_id'], x['stars']))
    ).to_dict()

    hits = 0
    ndcg_sum = 0
    num_users = 0

    for user_id, true_items in test_dict.items():
        if user_id not in predictions:
            continue

        num_users += 1
        user_recs = predictions[user_id][:k]

        hit = any(item in true_items for item in user_recs)
        if hit:
            hits += 1

        dcg = 0
        idcg = 0

        #CALCULATING REAL DCG
        for i, item in enumerate(user_recs):
            if item in true_items:
                relevance_score = true_items[item]
                dcg += relevance_score / math.log2(i + 2)

        #CALCULATING IDEAL DCG
        ideal_relevance_scores = sorted(true_items.values(), reverse=True)
        num_ideal_hits = min(len(ideal_relevance_scores), k)

        for i in range(num_ideal_hits):
            idcg += ideal_relevance_scores[i] / math.log2(i + 2)

        if idcg > 0: #avoid dividing by 0
            ndcg_sum += dcg / idcg

    if num_users == 0:
        return {"Hit Rate": 0.0, "NDCG": 0.0}

    return {
        f"Hit@{k}": hits/num_users,
        f"NDCG@{k}": ndcg_sum/num_users
    }


def evaluate_rating_metrics(true_stars, predicted_stars):
    mae = mean_absolute_error(true_stars, predicted_stars)
    mse = mean_squared_error(true_stars, predicted_stars)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "RMSE": rmse}