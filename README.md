# CID — Counterfactual-based Attribution

A modular Python library for constructing based on the paper CID: Measuring Feature Importance Through Counterfactuals Distributions. Link to paper: https://proceedings.mlr.press/v307/conti26a

## Core idea

CID follows a common attribution strategy:

```text
Instance
   ↓
Generate two sets of counterfactual points
   ↓
Infer the distributions
   ↓
Measure their dissimilarity
   ↓
Feature attribution
```

For each feature, the method compares its distribution in two sets of counterfactuals:

* points belonging to the **predicted class**;
* points belonging to the **opposite class**.

The resulting dissimilarity is used as the feature attribution.

The framework is modular: changing any of the three components results in a different attribution method, while preserving the same underlying strategy.

## Three components

### 1. Counterfactual generation

Generates the two sets of counterfactual points.

Required interface:

```python
cf_function(instance, amount_of_cfs)
```

Must return:

```python
predicted_class_data, opposite_class_data
```
Intializer.py contains DICE, KNeighbors and RANDOM
### 2. Distribution approximation

Infers or approximates the two distributions from the counterfactual samples.

Required interface:

```python
distr_approx(set_1, set_2)
```

Must return:

```python
distribution_1, distribution_2, points
```

Dissimilarity_Measure.py include KDE and ECDF.

### 3. Distribution dissimilarity

Measures the dissimilarity between the two distributions.

Required interface:

```python
dist_function(distribution_1, distribution_2, points)
```

Must return a **single scalar value**.

Dissimilarity_Measure.py include continuous Jaccard and Wasserstein distance.

## Custom functions

All three components can be replaced by user-defined functions.

For example:

```python
def my_distribution(set_1, set_2):
    # Your distribution approximation
    ...
    return distribution_1, distribution_2, points
```

and:

```python
def my_distance(distribution_1, distribution_2, points):
    # Your dissimilarity measure
    ...
    return distance
```

They can then be passed directly to the explainer:

```python
explainer = CIDNumericalExplainer(
    training_data=training_data,
    features_names=features_names,
    model=model,
    cf_function=my_cf_function,
    distr_approx=my_distribution,
    dist_function=my_distance
)
```

Functions can also be implemented as **callable classes** using `__call__`, which is useful when the method requires parameters that should be configured once and reused.

The only requirement is that the custom component respects the corresponding interface and output format described above.

