from argparse import ArgumentParser
from collections import defaultdict
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import surprise
from surprise.accuracy import rmse
from surprise.dump import dump, load
from surprise.model_selection import GridSearchCV

SAVED_MODELS_DIR = 'saved_models'

ALL_AGG_PREDICTIONS_STRATEGIES = ('Average', 'Average Without Misery', 'Multiplicative', 'Least Misery',
    'Most Pleasure', 'Approval Voting', 'Borda Count')
AGG_PREDICTIONS_STRATEGIES_MAINTAINING_SCALE = {'Average', 'Average Without Misery', 'Least Misery',
    'Most Pleasure'}

MODELS_HYPERPARAMS = {
    'SVD': {
        'n_factors': [50, 100, 200],
        'n_epochs': [20, 30, 40],
        'lr_all': [0.0001, 0.005, 0.01],
        'reg_all': [0.02, 0.04, 0.1]
    },
    'KNNBaseline': {
        'k': None, # To be defined given the selected group aggregation strategy
        'sim_options': { 
            'name': ['pearson_baseline'],
            'user_based': [True],
            'min_support': [2, 5, 10],
            'shrinkage': [20, 50, 100]
        },
        'bsl_options': {
            'method': ['als'],
            'reg_i': [10],
            'reg_u': [15],
            'n_epochs': [10, 20]
        },
        'verbose': [False]
    }
}

def init_hyperparam_grid(model_name, agg_method):
    """Initiate empty hyperparameter values in the MODELS_HYPERPARAMS dict.
    
    Parameters:
        model_name (str): class name corresponding to a prediction algorithm defined in the surprise package.
        agg_method (str): whether we are following the Aggregated Models ('agg-models') or the Aggregated
            Predictions ('agg-predictions') strategy for group recommendations.
    """
    if model_name == 'KNNBaseline':
        init_knnbaseline_hyperparamparam_grid(agg_method)

def init_knnbaseline_hyperparamparam_grid(agg_method):
    """Initiate empty hyperparameter values specific to the KNNBaseline prediction algorithm in the
    MODELS_HYPERPARAMS dict.
    
    Parameters:
        agg_method (str): whether we are following the Aggregated Models ('agg-models') or the Aggregated
            Predictions ('agg-predictions') strategy for group recommendations.
    """
    if agg_method == 'agg-models':
        MODELS_HYPERPARAMS['KNNBaseline']['k'] = [3, 5, 7]
    else:
        MODELS_HYPERPARAMS['KNNBaseline']['k'] = [10, 20, 30]

def load_dataset():
    """Loads the MovieLens (100k) ratings dataset and drops the timestamp column.
    
    Returns:
        pd.DataFrame: ratings dataset.
    """
    df = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
    return df.drop(['timestamp'], axis=1)

def map_users_to_groups(df, group_size, seed):
    """Divides the provided users in groups of a certain size.
    numpy.array_split (https://numpy.org/doc/stable/reference/generated/numpy.array_split.html) allows us
    to evenly distribute users when the remainder of num_users / group_size is not 0

    Parameters:
        df (pd.DataFrame): original ratings dataset.
        group_size (int): number of users in each group.
        seed (int): used for randomly assigning users to each group.
        
    Returns:
        dict: maps each user ID to a group ID.
    """
    user_ids = df.sample(frac=1, random_state=seed)['userId'].unique().tolist()
    num_groups = len(user_ids) // group_size
    groups = np.array_split(user_ids, num_groups)
    user_group_map = {}
    for i, group in enumerate(groups, start=1):
        for user in group:
            user_group_map[user] = i
    return user_group_map

def get_precision_and_recall_at_k(predictions, k, threshold):
    """Return precision and recall at k metrics for each group.
    
    Extracted from https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
    
    How precision is computed has been modified, since the original code filters the k recommended items
    using the provided threshold; instead, we assume that we will always recommend k items, even if the
    predicted rating for some of them falls below the threshold.

    Parameters:
        predictions (list[surprise.prediction_algorithms.predictions.Prediction])): named tuple of predictions
            as returned by the surprise package:
            (https://github.com/NicolasHug/Surprise/blob/2381fb11d0c4bf917cc4b9126f205d0013649966/surprise/prediction_algorithms/predictions.py#L21):
                uid (str): user id (it actually corresponds to a group id).
                iid (str): item id (not used).
                r_ui (float): true rating.
                est (float): estimated rating.
                details (dict): additional details about the prediction (not used).
        k (int): number of items to recommend.
        threshold (float): used to determine which items are relevant for group given the rating.

    Returns:
        float: Precision@k
        float: Recall@k
    """

    # First map the predictions to each group.
    group_est_true = defaultdict(list)
    for gid, _, true_r, est, _ in predictions:
        group_est_true[gid].append((est, true_r))

    precisions = []
    recalls = []
    for gid, group_ratings in group_est_true.items():
        # Sort group ratings by estimated value
        group_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in group_ratings)

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            (true_r >= threshold)
            for (_, true_r) in group_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        precisions.append(n_rel_and_rec_k / k)

        # Original code also filter the k recommended items using the threshold (so it may be
        # less than k). We assume that we always recommend the top k items. Uncomment the
        # following lines if you want to follow that other approach.

        # # Number of relevant and recommended items in top k
        # n_rel_and_rec_k = sum(
        #     ((true_r >= threshold) and (est >= threshold))
        #     for (est, true_r) in group_ratings[:k]
        # )
        # # Number of recommended items in top k
        # n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        # precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

    return np.mean(precisions), np.mean(recalls)

def get_coverage(predictions, threshold):
    """Return group coverage as the share of groups for whom at least one item can be recommended
    (in other words, there's at least one relevant recommended item). It depends on the provided
    threshold.

    Parameters:
        predictions (list[surprise.prediction_algorithms.predictions.Prediction])): named tuple of predictions
            as returned by the surprise package:
                uid (str): user id (it actually corresponds to a group id).
                iid (str): item id (not used).
                r_ui (float): true rating (not used).
                est (float): estimated rating.
                details (dict): additional details about the prediction (not used).
        threshold (float): used to determine which items are relevant for the group given the rating.

    Returns:
        float: coverage
    """

    # First map the predictions to each group.
    group_est_true = defaultdict(list)
    for gid, _, _, est, _ in predictions:
        group_est_true[gid].append(est)

    groups_with_prediction = 0
    groups = 0
    for gid, group_ratings in group_est_true.items():
        # Sort group ratings by estimated value
        group_ratings.sort(key=lambda x: x, reverse=True)

        if group_ratings[0] >= threshold:
            groups_with_prediction += 1
        groups += 1

    return groups_with_prediction / groups

def get_rmse_on_user_ratings(user_ratings_df, predictions):
    """Computes the RMSE between the predicted group ratings and the original ratings for each item by 
    an individual user belonging to that group.
    
    Parameters:
        user_ratings_df (pd.DataFrame): original ratings dataset containing the groupId assigned to each userId.
        predictions (list[surprise.prediction_algorithms.predictions.Prediction])): named tuple of predictions
            as returned by the surprise package:
                uid (str): group id.
                iid (str): item id.
                r_ui (float): true rating for the corresponding group (not used).
                est (float): estimated rating.
                details (dict): additional details about the prediction (not used).

    Returns:
        float: RMSE between predictions for groups and the original predictions of each user assigned to the groups
    """
    user_ratings_df = user_ratings_df.copy()
    user_ratings_df['predicted'] = np.nan
    for (group_id, item_id, _, est, _) in predictions:
        query = (user_ratings_df['groupId'] == int(group_id)) & (user_ratings_df['movieId'] == int(item_id))
        user_ratings_df.loc[query, 'predicted'] = est
    
    # Filter those items that have not been predicted
    predictions_df = user_ratings_df.loc[user_ratings_df['predicted'].notnull(),]
    return np.sqrt(np.sum((predictions_df['rating'] - predictions_df['predicted']) ** 2) / predictions_df.shape[0])

def compute_agg_models_metrics(df, predictions, num_predicted_items, relevance_threshold):
    """Compute metrics on the predictions obtained using the Aggregated Models approach.
    
    Parameters:
        df (pd.DataFrame): original ratings dataset.
        predictions (list[surprise.prediction_algorithms.predictions.Prediction])): named tuple of predictions
            as returned by the surprise package:
                uid (str): group id.
                iid (str): item id.
                r_ui (float): true rating for the corresponding group.
                est (float): estimated rating.
                details (dict): additional details about the prediction.
        num_predicted_items (int): number of items to recommend to each group.
        relevance_threshold (float): threshold used to check whether an item is relevant for a group or not.
            Must be in the [0.5, 5.0] range.

    Returns:
        dict: obtained value for each metric.
    """
    rmse_group_ratings = rmse(predictions, verbose=False)
    print(f' - RMSE on aggregated group ratings: {rmse_group_ratings}')
    
    rmse_user_ratings = get_rmse_on_user_ratings(df, predictions)
    print(f' - RMSE on original user ratings: {rmse_user_ratings}')

    precision_at_k, recall_at_k = get_precision_and_recall_at_k(predictions, num_predicted_items,
                                                                relevance_threshold)
    print(f' - Precision@{num_predicted_items}: {precision_at_k}')
    print(f' - Recall@{num_predicted_items}: {recall_at_k}')

    coverage = get_coverage(predictions, relevance_threshold)
    print(f' - Coverage: {coverage}')

    metrics = {
        'RMSE-group-ratings': rmse_group_ratings,
        'RMSE-user-ratings': rmse_user_ratings,
        'Precision@k': precision_at_k,
        'Recall@k': recall_at_k,
        'Coverage': coverage
    }
    return metrics

def get_agg_model_file_path(model_name, group_size, train_size, seed):
    """Creates the file path for a model trained using the Aggregated Models approach."""
    file_name = f'{model_name}_agg-models_g{group_size}_t{str(train_size).replace('.', '-')}_s{seed}'
    return os.path.join(SAVED_MODELS_DIR, file_name)

def train_agg_models_model(model_name, train_size, group_size, num_predicted_items, relevance_threshold, seed,
                           skip_training=False, save_model=True):
    """Trains a Group Recommender System following the Aggregated Models approach.
    Group profiles are created using the average rating given by the users for each item.
    Performs hyperparameter tuning using Grid Search.

    Can also be used just to load an already trained model and computing metrics on the same test
    holdout partition if skip_training is True.
    
    Parameters:
        model_name (str): class name corresponding to a prediction algorithm defined in the surprise package.
        train_size (float): proportion of instances (ratings) to be included into the train dataset in a
            train-test holdout.
        group_size (int): number of users in each group.
        num_predicted_items (int): number of items to recommend to each group.
        relevance_threshold (float): threshold used to check whether an item is relevant for a group or not.
            Must be in the [0.5, 5.0] range.
        seed (int): used for assigning users to groups, partitioning the data and in order to reproduce the
            training/hyperparameter search process.
        skip_training (bool): if True, will try to load an existing model and compute the metrics on the test
            dataset (model will not be saved again even if save_model is True).
        save_model (bool): whether to save the model parameters in the file system.

    Returns:
        dict: obtained value for each metric on the test dataset (best model).

    Raises:
        FileNotFoundError: when skip_training is True and the model cannot be loaded (it may no have been
            trained yet).
    """
    print('Selected approach: Aggregated Models')

    # 1. Load dataset
    df = load_dataset()
    
    # 2. Assign users to groups (first shuffle the user ids with the sample method)
    user_group_map = map_users_to_groups(df, group_size, seed)
    
    # 3. Aggregate users into group profiles using the average rating for each item
    df['groupId'] = df['userId'].map(user_group_map)
    agg_df = df.groupby(['groupId', 'movieId'])['rating'].mean().reset_index()

    # 4. Create train and test datasets
    trainset, testset = split_train_test(agg_df, train_size, seed)

    # 5. Train the model or load an already trained model
    if skip_training:
        file_path = get_agg_model_file_path(model_name, group_size, train_size, seed)
        _, best_model = load(file_path)
    else:
        np.random.seed(seed)
        init_hyperparam_grid(model_name, 'agg-models')
        model_class = getattr(surprise, model_name)  # dynamic import of model class

        print('Training...')
        start = time.time()
        gs = GridSearchCV(model_class, MODELS_HYPERPARAMS[model_name], measures=['rmse'], cv=5)
        gs.fit(trainset)
        end = time.time()
        print(f"Best params: {gs.best_params['rmse']}, RMSE score: {gs.best_score['rmse']}")
        print(f'Elapsed time: {end-start:.3f} seconds')
        best_model = gs.best_estimator['rmse']
        best_model.fit(trainset.build_full_trainset())

    # 6. Evaluate the best model on the test dataset
    predictions = best_model.test(testset)
    print('Evaluation of best model on test dataset:')
    metrics = compute_agg_models_metrics(df, predictions, num_predicted_items, relevance_threshold)

    # 7. Save the model if requested and training was done
    if save_model and not skip_training:
        print('Saving best model...')
        saved_models_path = Path(SAVED_MODELS_DIR)
        saved_models_path.mkdir(exist_ok=True)
        file_path = get_agg_model_file_path(model_name, group_size, train_size, seed)
        dump(file_path, algo=best_model)

    return metrics
    
def split_train_test(df, train_size, seed):
    """Performs a holdout partition of the given dataset.
    
    Extracted from https://github.com/NicolasHug/Surprise/blob/master/examples/split_data_for_unbiased_estimation.py

    Parameters:
        df (pd.DataFrame): ratings dataset.
        train_size (float): proportion of instances (ratings) to be included into the train dataset in a
            train-test holdout.
        seed (int): used to shuffle and partition in a reproducible manner.
            
    Returns:
        surprise.DatasetAutoFolds: train dataset.
        list[tuple[str, str, float]]: test dataset (user id, item id, true rating).
    """
    reader = surprise.Reader(rating_scale=(0.5, 5))
    df = df.sample(frac=1, random_state=seed) # shuffle
    data = surprise.Dataset.load_from_df(df, reader)
    raw_ratings = data.raw_ratings
    threshold = int(train_size * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]
    data.raw_ratings = train_raw_ratings
    testset = data.construct_testset(test_raw_ratings)
    return data, testset

def get_agg_predictions_file_path(model_name, train_size, seed):
    """Creates the file path for a model trained using the Aggregated Predictions approach."""
    file_name = f'{model_name}_agg-predictions_t{str(train_size).replace('.', '-')}_s{seed}'
    return os.path.join(SAVED_MODELS_DIR, file_name)

def df_aggregated_predictions_to_tuple(df_aggregated_predictions):
    """Convert a DataFrame of aggregated group predictions into the
    surprise.prediction_algorithms.predictions.Prediction format."""
    aggregated_predictions = []
    for _, row in df_aggregated_predictions.iterrows():
        aggregated_predictions.append((row['groupId'], row['iid'], row['r_ui'], row['est'], None))
    return aggregated_predictions

def agg_predictions_average(df_predictions):
    """Aggregate predictions for each group using the average.
    Also averages the true ratings."""
    return df_predictions.groupby(['groupId', 'iid']).agg(
        {'r_ui': 'mean', 'est': 'mean'}).reset_index()

def agg_predictions_average_without_misery(df_predictions, threshold=2.5):
    """Aggregate predictions for each group using the average without missery approach.
    For each group and item, if any of the estimated values for the users in that group
    is below the provided threshold, then this item will not be returned for this group.
    The threshold defaults to half the maximum movie rating.
    Also averages the true ratings."""
    avg_without_missery = df_predictions.groupby(['groupId', 'iid']).agg(
        **{
            'r_ui': ('r_ui', 'mean'),
            'est': ('est', lambda x: np.nan if np.any(x < threshold) else np.mean(x))
        }).reset_index()
    return avg_without_missery.loc[avg_without_missery['est'].notnull(),]

def agg_predictions_multiplicative(df_predictions):
    """Aggregate predictions for each group using the multiplication of values.
    Also averages the true ratings."""
    return df_predictions.groupby(['groupId', 'iid']).agg(
        {'r_ui': 'mean', 'est': 'prod'}).reset_index()

def agg_predictions_least_misery(df_predictions):
    """Aggregate predictions for each group using the Least Misery approach (minimum rating
    for each item over all the users that are member of the group).
    Also averages the true ratings."""
    return df_predictions.groupby(['groupId', 'iid']).agg(
        {'r_ui': 'mean', 'est': 'min'}).reset_index()

def agg_predictions_most_pleasure(df_predictions):
    """Aggregate predictions for each group using the Most Pleasure approach (maximum rating
    for each item over all the users that are member of the group).
    Also averages the true ratings."""
    return df_predictions.groupby(['groupId', 'iid']).agg(
        {'r_ui': 'mean', 'est': 'max'}).reset_index()

def agg_predictions_approval_voting(df_predictions, threshold=2.5):
    """Aggregate predictions for each group using the Approval Voting approach (number of predicted
    ratings that are above the specified threshold).
    Also averages the true ratings."""
    return df_predictions.groupby(['groupId', 'iid']).agg(
        **{
            'r_ui': ('r_ui', 'mean'),
            'est': ('est', lambda x: np.sum(x >= threshold))
        }).reset_index()

def agg_predictions_borda_count(df_predictions):
    """Aggregate predictions for each group using the Borda Count approach (sum of the rank that
    each item represents withing each user's ranking in ascending order of predicted rating).
    Also averages the true ratings."""
    df_predictions = df_predictions.copy()
    # First, we compute the ranking of items of each user given the predicted values
    df_predictions['user_ranking'] = np.nan
    for user_id in df_predictions['uid'].unique():
        user_ranking = df_predictions.loc[df_predictions['uid'] == user_id, 'est'].rank(ascending=True)
        df_predictions.loc[df_predictions['uid'] == user_id, 'user_ranking'] = user_ranking
    # Then, we sum all the group's user rankings for each item
    return df_predictions.groupby(['groupId', 'iid']).agg(
        **{
            'r_ui': ('r_ui', 'mean'),
            'est': ('user_ranking', 'sum')
        }).reset_index()

def create_agg_predictions(strategy, df_predictions):
    """Applies the specified aggregation strategy for groups on the provided predictions.
    
    Parameters:
        df_predictions (pd.DataFrame): predictions on the rating value of each item for individidual users.

    Returns:
        pd.DataFrame: aggregated predictions for each group and item following the specified strategy.
    """
    if strategy == 'Average':
        return agg_predictions_average(df_predictions)
    if strategy == 'Average Without Misery':
        return agg_predictions_average_without_misery(df_predictions)
    if strategy == 'Multiplicative':
        return agg_predictions_multiplicative(df_predictions)
    if strategy == 'Least Misery':
        return agg_predictions_least_misery(df_predictions)
    if strategy == 'Most Pleasure':
        return agg_predictions_most_pleasure(df_predictions)
    if strategy == 'Approval Voting':
        return agg_predictions_approval_voting(df_predictions)
    if strategy == 'Borda Count':
        return agg_predictions_borda_count(df_predictions)

def compute_agg_predictions_metrics(user_group_map, predictions, num_predicted_items, relevance_threshold):
    """Compute metrics on the predictions obtained using the Aggregated Predictions approach.
    
    Parameters:
        user_group_map (dict[int, int]): contains the group each user belongs to.
        predictions (list[surprise.prediction_algorithms.predictions.Prediction])): named tuple of predictions
            as returned by the surprise package:
                uid (str): group id.
                iid (str): item id.
                r_ui (float): true rating for the corresponding group.
                est (float): estimated rating.
                details (dict): additional details about the prediction.
        num_predicted_items (int): number of items to recommend to each group.
        relevance_threshold (float): threshold used to check whether an item is relevant for a group or not.
            Must be in the [0.5, 5.0] range.

    Returns:
        dict: obtained value for each metric.
    """
    rmse_user_ratings = rmse(predictions, verbose=False)
    print(f' - RMSE on original user ratings: {rmse_user_ratings}')

    metrics = {
        'RMSE-user-ratings': rmse_user_ratings
    }

    # Include the groupId in the predictions dataframe
    df_predictions = pd.DataFrame(predictions).drop(['details'], axis=1)
    df_predictions['groupId'] = df_predictions['uid'].map(user_group_map)

    all_agg_strategy_metrics = {}
    for strategy in ALL_AGG_PREDICTIONS_STRATEGIES:
        agg_strategy_metrics = {}
        print(f'\tStrategy: {strategy}')
        df_agg_predictions = create_agg_predictions(strategy, df_predictions)
        agg_predictions = df_aggregated_predictions_to_tuple(df_agg_predictions)

        if strategy in AGG_PREDICTIONS_STRATEGIES_MAINTAINING_SCALE:
            rmse_group_ratings = rmse(agg_predictions, verbose=False)
            print(f'\t - RMSE on aggregated group ratings (Average): {rmse_group_ratings}')
            
            coverage = get_coverage(agg_predictions, relevance_threshold)
            print(f'\t - Coverage: {coverage}')

            agg_strategy_metrics['RMSE-group-ratings'] = rmse_group_ratings
            agg_strategy_metrics['Coverage'] = coverage

        precision_at_k, recall_at_k = get_precision_and_recall_at_k(agg_predictions, num_predicted_items,
                                                                    relevance_threshold)
        print(f'\t - Precision@{num_predicted_items}: {precision_at_k}')
        print(f'\t - Recall@{num_predicted_items}: {recall_at_k}\n')

        agg_strategy_metrics['Precision@k'] = precision_at_k
        agg_strategy_metrics['Recall@k'] = recall_at_k

        all_agg_strategy_metrics[strategy] = agg_strategy_metrics
    
    metrics['by-strategy'] = all_agg_strategy_metrics
    return metrics

def train_agg_predictions_model(model_name, train_size, group_size, num_predicted_items, relevance_threshold, seed,
                                skip_training=False, save_model=True):
    """Trains a Group Recommender System following the Aggregated Predictions approach.
    Performs hyperparameter tuning using Grid Search. Computes metrics for every aggregation
    strategy.

    Can also be used just to load an already trained model and computing metrics on the same test
    holdout partition if skip_training is True.
    
    Parameters:
        model_name (str): class name corresponding to a prediction algorithm defined in the surprise package.
        train_size (float): proportion of instances (ratings) to be included into the train dataset in a
            train-test holdout.
        group_size (int): number of users in each group.
        num_predicted_items (int): number of items to recommend to each group.
        relevance_threshold (float): threshold used to check whether an item is relevant for a group or not.
            Must be in the [0.5, 5.0] range.
        seed (int): used for assigning users to groups, partitioning the data and in order to reproduce the
            training/hyperparameter search process.
        skip_training (bool): if True, will try to load an existing model and compute the metrics on the test
            dataset (model will not be saved again even if save_model is True).
        save_model (bool): whether to save the model parameters in the file system.

    Returns:
        dict: obtained value for each metric on the test dataset (best model).

    Raises:
        FileNotFoundError: when skip_training is True and the model cannot be loaded (it may no have been
            trained yet).
    """
    print('Selected approach: Aggregated Predictions')
    
    # 1. Load dataset
    df = load_dataset()
    
    # 2. Assign users to groups (first shuffle the user ids with the sample method)
    user_group_map = map_users_to_groups(df, group_size, seed)
    
    # 3. Include groupId in the dataset according to the obtained mapping 
    df['groupId'] = df['userId'].map(user_group_map)

    # 4. Create train and test datasets
    trainset, testset = split_train_test(df[['userId', 'movieId', 'rating']], train_size, seed)
    
    # 5. Train the model or load an already trained model
    if skip_training:
        file_path = get_agg_predictions_file_path(model_name, train_size, seed)
        _, best_model = load(file_path)
    else:
        np.random.seed(seed)
        init_hyperparam_grid(model_name, 'agg-predictions')
        model_class = getattr(surprise, model_name)  # dynamic import of model class

        print('Training...')
        start = time.time()
        gs = GridSearchCV(model_class, MODELS_HYPERPARAMS[model_name], measures=['rmse'], cv=5)
        gs.fit(trainset)
        end = time.time()
        print(f"Best params: {gs.best_params['rmse']}, RMSE score: {gs.best_score['rmse']}")
        print(f'Elapsed time: {end-start:.3f} seconds')
        best_model = gs.best_estimator['rmse']
        best_model.fit(trainset.build_full_trainset())

    # 6. Evaluate the best model on the test dataset
    predictions = best_model.test(testset)
    print('Evaluation of best model on test dataset:')
    metrics = compute_agg_predictions_metrics(user_group_map, predictions, num_predicted_items, relevance_threshold)

    # 7. Save the model if requested and training was done
    if save_model and not skip_training:
        print('Saving best model...')
        saved_models_path = Path(SAVED_MODELS_DIR)
        saved_models_path.mkdir(exist_ok=True)
        file_path = get_agg_predictions_file_path(model_name, train_size, seed)
        dump(file_path, algo=best_model)

    return metrics

def validate_arguments(args):
    """Validates the value of the provided command line arguments."""
    if not(0 < args.train_size < 1):
        raise ValueError('Provided train size is not valid (must be greater than 0 and smaller than 1)')
    if not(3 <= args.group_size <= 10):
        raise ValueError('Provided group size is not valid (given the number of users in MovieLens 100K, '
                         'must be smaller than or equal to 10)')
    if args.num_predicted_items < 1:
        raise ValueError('Provided value for the number of predicted items for each group is not valid')
    if not(0.5 <= args.relevance_threshold <= 5):
        raise ValueError('Provided relevance threshold is not valid for the MovieLens dataset')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['SVD', 'KNNBaseline'], default='SVD')
    parser.add_argument('-a', '--agg-method', choices=['agg-models', 'agg-predictions'], required=True)
    parser.add_argument('-t', '--train-size', type=float, default=0.75)
    parser.add_argument('-g', '--group-size', type=int, default=5)
    parser.add_argument('-k', '--num-predicted-items', type=int, default=10)
    parser.add_argument('-r', '--relevance-threshold', type=float, default=4)
    parser.add_argument('-s', '--seed', type=int, default=714)
    parser.add_argument('-n', '--skip-training', default=False, action='store_true')
    args = parser.parse_args()

    if args.agg_method == 'agg-models':
        train_agg_models_model(args.model, args.train_size, args.group_size, args.num_predicted_items,
                               args.relevance_threshold, args.seed, args.skip_training)
    else:
        train_agg_predictions_model(args.model, args.train_size, args.group_size, args.num_predicted_items,
                                    args.relevance_threshold, args.seed, args.skip_training)
    