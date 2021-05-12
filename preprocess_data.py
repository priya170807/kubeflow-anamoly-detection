import pandas as pd
import config
df = pd.read_csv(config.data_path)
# we don't have any categorical data to preprocess and there are no Nan values in te dataset
# upload the data into the google cloud bucket
df.to_csv("./data_processed.csv")
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(config.gs_bucket_name)
bucket.blob('processed/data_processed.csv').upload_from_filename('./data_processed.csv', content_type='text/csv')
print("Raw Data Processed Sucessfully")
