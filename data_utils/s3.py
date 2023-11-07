# Import s3fs library for accessing S3 files
import s3fs

# Define the access key and secret key for the S3 service
access_key = ""
secret_key = ""

# Define the endpoint for the S3 service
endpoint = "eodata.dataspace.copernicus.eu"

# Define the bucket name for the S3 service
bucket = "s5p"

# Define the prefix for the product ID and time range
prefix = "L2__O3_TCL/2020/01"

# Create a S3 file system object with the credentials and endpoint
s3 = s3fs.S3FileSystem(key=access_key, secret=secret_key, client_kwargs={"endpoint_url": "https://" + endpoint})

# List the files in the bucket with the prefix
files = s3.ls(bucket + "/" + prefix)

# Loop through the files and download them to the local directory
for file in files:
    # Get the file name from the file path
    file_name = file.split("/")[-1]
    # Download the file to the local directory
    s3.get(file, file_name)
    # Print the file name and size
    print("Downloaded", file_name, "with size", s3.size(file), "bytes")
