import pandas as pd
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def load_data():
    train_df = pd.read_csv('train_reviews_santa_barbara.csv')
    test_df = pd.read_csv('test_reviews_santa_barbara.csv')
    restaurants_df = pd.read_csv('restaurants_santa_barbara.csv')
    return train_df, test_df, restaurants_df


def evaluate_model(predictions, test_df, k_list=[10, 20, 30]):
    test_dict = test_df.groupby('user_id').apply(
        lambda x: dict(zip(x['business_id'], x['stars']))
    ).to_dict()

    results = {f"Hit@{k}": 0.0 for k in k_list}
    results.update({f"NDCG@{k}": 0.0 for k in k_list})
    num_users = 0

    for user_id, true_items in test_dict.items():
        if user_id not in predictions:
            continue

        num_users += 1
        user_recs_all = predictions[user_id]

        for k in k_list:
            user_recs = user_recs_all[:k]

            hit = any(item in true_items for item in user_recs)
            if hit:
                results[f"Hit@{k}"] += 1

            dcg = 0
            idcg = 0

            # CALCULATING REAL DCG
            for i, item in enumerate(user_recs):
                if item in true_items:
                    relevance_score = true_items[item]
                    dcg += relevance_score / math.log2(i + 2)

            # CALCULATING IDEAL DCG
            ideal_relevance_scores = sorted(true_items.values(), reverse=True)
            num_ideal_hits = min(len(ideal_relevance_scores), k)

            for i in range(num_ideal_hits):
                idcg += ideal_relevance_scores[i] / math.log2(i + 2)

            if idcg > 0:  # avoid dividing by 0
                results[f"NDCG@{k}"] += dcg / idcg

    if num_users == 0:
        return {metric: 0.0 for metric in results}

    for metric in results:
        results[metric] /= num_users

    return results


def evaluate_rating_metrics(true_stars, predicted_stars):
    mae = mean_absolute_error(true_stars, predicted_stars)
    mse = mean_squared_error(true_stars, predicted_stars)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_stars, predicted_stars)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}