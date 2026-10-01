import pandas as pd

df = pd.read_csv("dataset_inventory.csv")

article_table = df[
    [
        "dataset",
        "rows",
        "target_unique",
        "boolean_features_est",
        "categorical_features_est",
        "numerical_features_est",
        "missing_values",
    ]
].copy()

article_table.columns = [
    "Dataset",
    "Instances",
    "Classes",
    "Boolean",
    "Categorical",
    "Numerical",
    "Missing Values",
]

article_table["Features"] = (
    article_table["Boolean"]
    + article_table["Categorical"]
    + article_table["Numerical"]
)

article_table = article_table[
    [
        "Dataset",
        "Instances",
        "Features",
        "Classes",
        "Boolean",
        "Categorical",
        "Numerical",
        "Missing Values",
    ]
]

article_table["Dataset"] = (
    article_table["Dataset"]
    .str.replace("_", " ", regex=False)
    .str.replace("-", " ", regex=False)
    .str.title()
)

article_table = article_table.sort_values("Dataset")

article_table.to_csv("table_datasets_for_article_corrected.csv", index=False)
article_table.to_excel("table_datasets_for_article_corrected.xlsx", index=False)

print(article_table)