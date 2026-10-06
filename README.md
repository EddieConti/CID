# CID — Counterfactual-based Attribution

A modular Python library for feature attribution based on the paper *CID: Measuring Feature Importance Through Counterfactuals Distributions*. Link to paper: https://proceedings.mlr.press/v307/conti26a

## Core idea

CID follows a common attribution strategy:

```text
Instance
   ↓
Generate two sets of counterfactual points (Recommended: KNeighbors)
   ↓
Infer the distributions (Recommended: _ecdf)
   ↓
Measure their dissimilarity (Recommended: _wasserstein)
   ↓
Feature attribution
```

For each feature, the method compares its distribution in two sets of counterfactuals:

* points belonging to the **predicted class**;
* points belonging to the **opposite class**.

The resulting dissimilarity is used as the feature attribution. The framework is modular: changing any of the components results in a different attribution method, while preserving the same underlying strategy.

## Structure

| File | Content |
|---|---|
| `CID_base.py` | `BaseCIDExplainer`: logic shared by both explainers (local and global explanations, plots) |
| `CID_numerical.py` | `CIDNumericalExplainer`: importance of the numerical features |
| `CID_categorical.py` | `CIDCategoricalExplainer`: importance of the categorical features |
| `Initializer.py` | Counterfactual generators: `DICE`, `KNeighbors`, `RANDOM` |
| `Dissimilarity_Measure.py` | Distribution approximations (`KDE`, `ECDF`) and dissimilarities (continuous Jaccard, Wasserstein, Jaccard distance for categories) |

## Flexibility

The dataset can **mix numerical and categorical features**. Pass it once, then use each explainer to get the importances of the features of its own type: the other features are kept fixed while generating counterfactuals.

* Features are detected from the dtypes (numeric vs. non-numeric). Integer-coded categoricals must be given explicitly with `features_names=[...]` (to both explainers).
* `model` must accept a DataFrame with all the columns except the target.
* `cf_function` can be `"knn"`, `"random"`, `"dice"` or any callable (see below).

```python
from CID_numerical import CIDNumericalExplainer
from CID_categorical import CIDCategoricalExplainer

num = CIDNumericalExplainer(training_data=df, target_ft_name="y", model=model)
cat = CIDCategoricalExplainer(training_data=df, target_ft_name="y", model=model)

# Local explanation
x = df.drop(columns="y").iloc[0]
num.explain_instance(x)
cat.explain_instance(x)

# Global explanation (mean over a sample of instances)
num.global_explanation(n_instances=30, variability=True)
cat.global_explanation(n_instances=30)
```

## Components

### 1. Counterfactual generation

```python
cf_function(instance, amount_of_cfs)  ->  predicted_class_data, opposite_class_data
```

It is important to have a function that generates two types of counterfactuals: instances with the same predicted class and instances with the opposite predicted class. DiCE is a counterfactual library that account for this possibility, we wrote other two methods: Kneighbors (recommended) and RANDOM. Both outputs are arrays containing only the explained features (columns in the order of `features_names`). Available in `Initializer.py`: `DICE`, `KNeighbors`, `RANDOM`.

### 2. Distribution approximation (numerical explainer)

```python
distr_approx(set_1, set_2)  ->  distribution_1, distribution_2, points
```

Available in `Dissimilarity_Measure.py`: `KDE`, `ECDF`.

### 3. Distribution dissimilarity

```python
dist_function(distribution_1, distribution_2, points)  ->  scalar          # numerical
dist_function(set_1, set_2)                            ->  scalar          # categorical
```

Available in `Dissimilarity_Measure.py`: continuous Jaccard and Wasserstein distance (numerical), Jaccard distance (categorical).

## Custom functions

Every component can be replaced by a user-defined function (or by a **callable class** with `__call__`, useful when parameters must be configured once and reused). The only requirement is to respect the interface above.

```python
def my_distribution(set_1, set_2):
    ...
    return distribution_1, distribution_2, points

def my_distance(distribution_1, distribution_2, points):
    ...
    return distance

explainer = CIDNumericalExplainer(
    training_data=df,
    target_ft_name="y",
    model=model,
    cf_function=my_cf_function,       # or "knn" / "random" / "dice"
    distr_approx=my_distribution,
    dist_function=my_distance,
)
```

## Examples

The `Examples` folder contains demo notebooks, including the categorical features case and multiclass classification.
