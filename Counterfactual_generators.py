import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import dice_ml
from sklearn.neighbors import NearestNeighbors



def _generate_dice(model,
    cf_explainer,
    instance,
    predicted_class,
    amount_of_cfs,
    training_data,
    features_names,
):
    if not isinstance(instance, pd.DataFrame):
        instance = instance.to_frame().T

    opposite_class = 1 - predicted_class

    opposite_class_cf = cf_explainer.generate_counterfactuals(
        instance,
        total_CFs=amount_of_cfs,
        desired_class=int(opposite_class),
    )

    predicted_class_cf = cf_explainer.generate_counterfactuals(
        instance,
        total_CFs=amount_of_cfs,
        desired_class=int(predicted_class),
    )

    opposite_class_data = (opposite_class_cf.cf_examples_list[0].final_cfs_df[features_names].values)

    predicted_class_data = (predicted_class_cf.cf_examples_list[0].final_cfs_df[features_names].values)

    return predicted_class_data, opposite_class_data


def _generate_neighbors(
    model,
    cf_explainer,
    instance,
    predicted_class,
    amount_of_cfs,
    training_data,
    features_names,
):
    _, indices = cf_explainer.kneighbors(instance[features_names])

    predicted_class_data = []
    opposite_class_data = []

    for index in indices[0]:
        candidate = training_data.iloc[index:index + 1]

        candidate_class = model.predict(candidate[features_names])[0]

        if candidate_class == predicted_class:
            predicted_class_data.append(candidate[features_names].squeeze())
        else:
            opposite_class_data.append(candidate[features_names].squeeze())

        if (len(predicted_class_data) >= amount_of_cfs and len(opposite_class_data) >= amount_of_cfs):
            break

    predicted_class_data = np.array(predicted_class_data[:amount_of_cfs])

    opposite_class_data = np.array(opposite_class_data[:amount_of_cfs])

    return predicted_class_data, opposite_class_data


def _generate_random(
    model,
    cf_explainer,
    instance,
    predicted_class,
    amount_of_cfs,
    training_data,
    features_names,
    n_samples=1000,
):
           

    data = np.random.normal(
            loc=instance[features_names].values,
            scale=cf_explainer.std().values,
            size=(n_samples, len(features_names)),
        )

    random_points = pd.DataFrame(data,columns=features_names)

    predictions = model.predict(random_points)

    predicted_class_data = random_points[predictions == predicted_class].values

    opposite_class_data = random_points[predictions != predicted_class].values

    if len(predicted_class_data) < amount_of_cfs:
        raise ValueError("Not enough same-class random samples.")

    if len(opposite_class_data) < amount_of_cfs:
        raise ValueError("Not enough opposite-class random samples.")

    return predicted_class_data[:amount_of_cfs], opposite_class_data[:amount_of_cfs]