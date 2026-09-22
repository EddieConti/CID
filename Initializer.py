import dice_ml
from sklearn.neighbors import NearestNeighbors

def _init_dice(model,training_data,features_names,target_ft_name,cf_generation_dice):
    data = dice_ml.Data(
            dataframe=training_data,
            continuous_features=features_names,
            outcome_name=target_ft_name,
        )
    
    dice_model = dice_ml.Model(
            model=model,
            backend="sklearn",
        )
    
    exp = dice_ml.Dice(
            data,
            dice_model,
            method=cf_generation_dice,
        )
    
    return exp


def _init_neighbors(
    model,
    training_data,
    features_names,
    target_ft_name,
    cf_generation_dice
):
    nbrs = NearestNeighbors(
        n_neighbors=len(training_data),
        algorithm="ball_tree"
    ).fit(training_data[features_names])

    return nbrs


def _init_random(
    model,
    training_data,
    features_names,
    target_ft_name,
    cf_generation_dice
):
    return training_data[features_names]
