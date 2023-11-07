# Import requests and datetime libraries
import requests
import datetime

# Define the username, password, and client ID for authentication
username = "j.w.miller1@lanaster.ac.uk"
password = "C0p3rn1cu$n3$$"
client_id = "sh-15f4cd06-810d-4615-aba2-d92fc21422ab"

OAuth_secret = ""

# Define the identity service URL and the products endpoint URL
identity_url = "https://auth.dataspace.copernicus.eu/auth/realms/Copernicus/protocol/openid-connect/token"
products_url = "https://dataspace.copernicus.eu/api/v1/catalog/1.0.0/collections/sentinel-5p-l2/products"

# Define the product type and the spatial extent
product_type = "L2__O3_TCL"
bbox = [-180, -90, 180, 90]

# Define the start and end dates
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 1, 31)

# Define a function to get an authorization token
def get_token():
    # Prepare the payload with the username, password, and client ID
    payload = {
        "username": username,
        "password": password,
        "client_id": client_id,
        "grant_type": "password"
    }
    # Send a POST request to the identity service with the payload
    response = requests.post(identity_url, data=payload)
    # Check if the response is successful
    if response.status_code == 200:
        # Parse the response as JSON and get the access token
        data = response.json()
        token = data["access_token"]
        # Return the token as a string
        return token
    else:
        # Raise an exception if the response is not successful
        raise Exception(f"Failed to get token: {response.text}")

# Define a function to download a product by ID
def download_product(product_id):
    # Get an authorization token
    token = get_token()
    # Construct the download URL by appending the product ID and $value to the products endpoint URL
    download_url = f"{products_url}/{product_id}/$value"
    # Prepare the headers with the authorization token
    headers = {
        "Authorization": f"Bearer {token}"
    }
    # Send a GET request to the download URL with the headers and stream the response content
    response = requests.get(download_url, headers=headers, stream=True)
    # Check if the response is successful
    if response.status_code == 200:
        # Open a file with the product ID as the name and write mode as binary
        with open(f"{product_id}.zip", "wb") as file:
            # Iterate over the response content chunks and write them to the file
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        # Print a message indicating that the download was successful
        print(f"Downloaded product {product_id}")
    else:
        # Raise an exception if the response is not successful
        raise Exception(f"Failed to download product {product_id}: {response.text}")

# Loop over each day from start date to end date (inclusive)
for date in (start_date + datetime.timedelta(days=n) for n in range((end_date - start_date).days + 1)):
    # Format the date as YYYY-MM-DD
    date_str = date.strftime("%Y-%m-%d")
    # Prepare the parameters for searching products by product type, date, and bbox
    params = {
        "$filter": f"productType eq '{product_type}' and beginPosition ge {date_str}T00:00:00Z and endPosition le {date_str}T23:59:59Z",
        "$select": "id",
        "$top": 100,
        "$skip": 0,
        "bbox": ",".join(map(str, bbox))
    }
    # Send a GET request to the products endpoint URL with the parameters
    response = requests.get(products_url, params=params)
    # Check if the response is successful
    if response.status_code == 200:
        # Parse the response as JSON and get the value list of products
        data = response.json()
        products = data["value"]
        # Loop over each product in the list
        for product in products:
            # Get the product ID from the product dictionary
            product_id = product["id"]
            # Download the product by ID using the download_product function
            download_product(product_id)
    else:
        # Raise an exception if the response is not successful
        raise Exception(f"Failed to search products for date {date_str}: {response.text}")
