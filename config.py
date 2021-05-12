gs_bucket_name="test-project-one-307022_bucket"
Bucket_uri="gs://test-project-one-307022_bucket"
version=1
store_artifacts=Bucket_uri + "/" + str(version)
data_path=Bucket_uri + "/" + "data/data_raw.csv"
processed_data=Bucket_uri + "/" + "processed/data_processed.csv"
