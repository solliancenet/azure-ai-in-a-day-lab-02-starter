import pandas as pd


# Loads the COVID articles sample data from dataset COVID19Articles_Test.
def create_sample_data_csv(file_name: str = "COVID19Articles.csv",
                           for_scoring: bool = False):

    url = \
        "https://solliancepublicdata.blob.core.windows.net" + \
        "/ai-in-a-day/lab-02/"
    df = pd.read_csv(url + file_name)
    if for_scoring:
        df = df.drop(columns=['cluster'])
    df.to_csv(file_name, index=False)
