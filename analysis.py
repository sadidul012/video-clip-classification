import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv("data/output.csv")
    print(df.head())
    print(df.label.unique())
    print(df.confidence.describe())
