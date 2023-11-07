import requests

# Define the base URL for the OData service
base_url = "https://scihub.copernicus.eu/dhus/odata/v1/"

# Define the filter option to query by product ID and time range
filter_option = "$filter=substringof('L2__O3_TCL',Name) and ContentDate ge datetime'2020-01-01T00:00:00.000Z' and ContentDate le datetime'2020-02-01T23:59:59.999Z'"

# Define the output format option to get the results as JSON
format_option = "$format=json"

# Construct the full URL by combining the base URL, filter option, and format option
full_url = base_url + "Products?" + filter_option + "&" + format_option

# Send a GET request to the full URL and get the response as JSON
response = requests.get(full_url).json()

# Print the number of products found
print("Number of products found:", response["d"]["__count"])

# Loop through the products and print their names and download URLs
for product in response["d"]["results"]:
    print("Name:", product["Name"])
    print("Download URL:", product["__metadata"]["media_src"])
    print()