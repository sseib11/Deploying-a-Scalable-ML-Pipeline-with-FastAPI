# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project trains a binary classification model to predict whether an individual’s income is greater than $50K per year (">50K") or less than or equal to $50K per year ("<=50K") using U.S. Census-style demographic and employment features. The model is a scikit-learn Logistic Regression classifier trained on tabular data. Categorical features are encoded using a OneHotEncoder and the target label is encoded using a LabelBinarizer. The trained model and preprocessing artifacts are saved to disk as `model/model.pkl`, `model/encoder.pkl`, and `model/lb.pkl`.

## Intended Use
The intended use of this model is educational: demonstrating an end-to-end machine learning pipeline (data processing, model training, evaluation, slicing, and deployment behind a FastAPI endpoint). The model can be used to learn how to serve a trained classifier and how to monitor performance across slices of data.

This model is not intended for real-world employment, lending, housing, insurance, or other high-stakes decision-making. Predictions should not be used to determine eligibility, benefits, or access to opportunities.

## Training Data
The training data is the provided `census.csv` dataset included in the starter repository. The dataset contains continuous and categorical demographic and employment-related fields (for example: age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country). The label is `salary`, representing whether income is ">50K" or "<=50K".

The training script performs an 80/20 train/test split and fits the OneHotEncoder and LabelBinarizer on the training subset.

## Evaluation Data
The evaluation data is the held-out 20% test split created by `train_model.py` using a fixed random seed. The same preprocessing steps (OneHotEncoder and LabelBinarizer fit on the training data) are applied to the test split, and metrics are computed on the test split predictions.

In addition to overall test metrics, slice metrics are computed by holding each categorical feature value fixed (for example, each distinct `workclass` value) and measuring precision, recall, and F1 on that subset. These slice results are written to `slice_output.txt`.

## Metrics
The model is evaluated using:
- **Precision**
- **Recall**
- **F1 score** (F-beta with beta = 1)

Overall performance on the held-out test dataset:
- **Precision:** 0.7289  
- **Recall:** 0.6071  
- **F1:** 0.6625  

Slice-based metrics are computed for each unique value of each categorical feature and saved in `slice_output.txt`. Example observations from the slice output include:
- Some categories have very small sample counts (for example, `workclass: Never-worked, Count: 1` and `marital-status: Married-AF-spouse, Count: 5`). Metrics on these very small slices are unstable and can appear as 0.0 or 1.0 due to low support.
- “Unknown/missing” category values represented as `?` appear in multiple fields (for example `workclass: ?` and `occupation: ?`), and slice performance for these groups differs from more common categories.

## Ethical Considerations
This dataset includes sensitive attributes (such as race and sex). Using these features in models can lead to unfair or discriminatory outcomes if the model is deployed in real decision-making contexts. Even if sensitive features are removed, other variables may act as proxies and still produce biased results.

Because income prediction models can easily be misused in high-impact settings, the model should not be used to make decisions that affect individuals’ rights or access to resources. If a similar model were to be deployed in practice, it would require careful bias assessment, fairness evaluation across groups, transparency requirements, and ongoing monitoring.

## Caveats and Recommendations
- The dataset is a simplified, historical snapshot and may not represent current populations or labor markets.
- Logistic Regression is a relatively simple model; performance could be improved with additional feature engineering, scaling of continuous features, hyperparameter tuning, or alternative model families. However, improved accuracy does not guarantee fairness.
- Slice metrics can be misleading for categories with very low counts. In practice, consider minimum-support thresholds, confidence intervals, or grouping rare categories.
- If extending beyond an educational setting, add stronger documentation, data validation, robust monitoring, and fairness testing before any real-world use.
