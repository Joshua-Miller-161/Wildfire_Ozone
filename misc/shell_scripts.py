import os
import re
import zipfile



def CombO3Dir(base_path, pattern=r"\(\d+\)", delete=False):
    # Create an empty list to store the matched files
    matched_files = []

    # Loop through the files in the directory
    for filename in os.listdir(base_path):
        # Check if the filename matches the pattern
        if re.search(pattern, filename):
            # Add the file to the matched list
            matched_files.append(filename)

    # Print the list of matched files
    print("The following files match the pattern:")
    for file in matched_files:
        print(file)

    if delete:
        # Delete the matched files
        for file in matched_files:
            # Get the full path of the file
            file_path = os.path.join(base_path, file)
            # Delete the file
            os.remove(file_path)

        # Print a confirmation message
        print("The matched files have been deleted.")


def UnZipAndDel(base_path):
    # Define a function that unzips a zip file
    def unzip_file(zip_file):
        # Get the full path of the zip file
        zip_path = os.path.join(base_path, zip_file)
        # Create a ZipFile object
        with zipfile.ZipFile(zip_path, "r") as zip_obj:
            # Extract all the contents of the zip file to the same base_path
            zip_obj.extractall(base_path)
            # Print a confirmation message
            print(f"{zip_file} has been unzipped.")

    zip_files = []

    for filename in os.listdir(base_path):
        print(filename[-4:])
        if (filename[-4:]=='.zip'):

            zip_files.append(os.path.join(base_path, filename))

            unzip_file(os.path.join(base_path, filename))
        
    return zip_files


def DelZip(base_path):
    for filename in os.listdir(base_path):
        if (filename[-4:] == '.zip'):
            os.remove(os.path.join(base_path, filename))


def Filter_OFFL_RPRO(base_path):
    filenames = os.listdir(base_path)
    filename.sort()

    for filename in filenames:
        

#CombO3Dir("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL", delete=True)

#zip_files = UnZipAndDel("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")

#DelZip("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")