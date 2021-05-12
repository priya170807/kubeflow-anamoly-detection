import os
import requests
import tensorflow as tf
import config
# Download the zipped dataset
url = 'https://fs-domain-bucket.s3.amazonaws.com/credit-card-fraud-detection/data-folder/creditcard.csv'
req = requests.get(url)
url_content = req.content
csv_file = open('./creditcard.csv', 'wb')
csv_file.write(url_content)
csv_file.close()

# upload file to google storage
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(config.gs_bucket_name)
bucket.blob('data/data_raw.csv').upload_from_filename('./creditcard.csv', content_type='text/csv')
print("Data DownLoaded Sucessfully")
