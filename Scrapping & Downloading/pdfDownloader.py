import os
import requests

# Read the contents of the text file
with open("links.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if("Category" in line):
            folder_name = line.split(' ')[-1]
            # Create the folder if it does not exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            continue
    
        # Get the filename from the end of the URL
        filename = os.path.basename(line)
        
        # Send a request to download the PDF file
        response = requests.get(line)
        if response.status_code == 200:
            print("File downloaded: ", filename)
        else:
            print("An error occurred while downloading file: ", filename)

        # Save the PDF file to the newly created folder
        with open(os.path.join(folder_name, filename), 'wb') as pdf_file:
            pdf_file.write(response.content)