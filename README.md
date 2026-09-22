# CID
Repository for the paper CID: Measuring Feature Importance Through Counterfactuals Distributions. Link to paper: https://proceedings.mlr.press/v307/conti26a
The library currently provides support for numerical features and binary/multiclass classification, with categorical feature support under development.

## Project Structure

The main components of the project are:

```text
.
├── CID_numerical.py
├── CID_categorical.py
├── initializer.py
├── Counterfactual_generators.py
├── Dissimilarity_measure.py
├── Introduction_Binary_Classification.ipynb
└── Introduction_Multiclass.ipynb
```

### `CID_numerical.py`

Contains the main `CIDNumericalExplainer` class for computing feature importance for numerical features.

The explainer follows a modular strategy in which counterfactual generation and the dissimilarity measure can be changed independently.

### `CID_categorical.py`

Contains the implementation for categorical features.

**Status:** under development.

### `initializer.py`

Contains the initialization strategies for the different counterfactual-generation methods.

The initializer is responsible for preparing the component required by the selected strategy.

### `Counterfactual_generators.py`

Contains the different counterfactual-generation strategies used by CID.

The modular structure allows different approaches to be used without changing the core CID algorithm.

### `Dissimilarity_measure.py`

Contains the distributional dissimilarity measure used by CID to compare the feature distributions of the generated counterfactual sets.

The dissimilarity measure is independent from the counterfactual-generation strategy.

## Examples

Two Jupyter notebooks are provided to demonstrate the use of the library:

### `Introduction_Binary_Classification.ipynb`

Example of CID applied to a **binary classification** problem.

### `Introduction_Multiclass.ipynb`

Example of CID applied to a **multiclass classification** problem.

The notebooks provide practical examples of how to initialize the explainer and obtain feature importance values.

## Basic Usage

A numerical CID explainer can be initialized as follows:

```python
from CID_numerical import CIDNumericalExplainer

explainer = CIDNumericalExplainer(
    training_data,
    model,
    target_ft_name="Outcome"
)

feature_importances = explainer.explain_instance(instance)
```

The resulting feature importance values correspond to the dissimilarity between the distributions associated with the different counterfactual classes for each feature.

## Modular Architecture

CID separates the main components of the method into independent modules:

```text
CIDNumericalExplainer
        │
        ├── Initializer
        │
        ├── Counterfactual Generator
        │
        └── Dissimilarity Measure
```

This structure makes it possible to experiment with different counterfactual-generation and dissimilarity strategies while keeping the main explainer unchanged.

