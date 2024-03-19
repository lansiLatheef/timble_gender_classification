import requests
import bs4
import json
import time
import concurrent.futures
import os
import sys

# Function to scrape data from URLs
def get_data(url):
    global full_sleeve_count
    print(url)
    res = requests.get('https://www.myntra.com/'+url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res.raise_for_status()
    except Exception as exc:
        print("There was a problem: %s" % (exc))
        return

    print('making soup...')
    soup_res = bs4.BeautifulSoup(res.text,'html.parser')

    scripts = soup_res.find_all('script')
    script_content = None
    for script in scripts:
        if script.string and 'pdpData' in script.string:
            script_content = script.string
            break

    if script_content:
        try:
            data_start_index = script_content.find('{')
            data_end_index = script_content.rfind('}') + 1
            data = json.loads(script_content[data_start_index:data_end_index])
        except json.JSONDecodeError as e:
            print("Error decoding JSON data:", e)
            return
    else:
        print("No script content containing 'pdpData' found.")
        return

    id = data["pdpData"]['id']
    product = data["pdpData"]['analytics']['articleType']
    gender = data["pdpData"]['analytics']["gender"]
    description = data["pdpData"]["name"]
    img1 = data["pdpData"]["media"]["albums"][0]["images"][0]["imageURL"]
    img2 = data["pdpData"]["media"]["albums"][0]["images"][1]["imageURL"]
    
    # Determine if the product is full sleeve or half sleeve
    if "full sleeve" in description.lower() or full_sleeve_count < 200:
        sleeve_type = "full sleeve"
        full_sleeve_count += 1
    elif "half sleeve" in description.lower():
        sleeve_type = "half sleeve"
    else:
        sleeve_type = "unknown"
    
    newdata = {
        'product': product,
        'img': img1,
        'gender': gender,
        'sleeve type': sleeve_type
    }

    final_data[id] = newdata
    
    print('collecting info...')        
    time.sleep(0.25)

# Function to process URLs concurrently
def get_url(links):
    threads = min(MAX_THREADS, len(links))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(get_data, links)

# Main function
def main():
    global full_sleeve_count
    # Reset full_sleeve_count
    full_sleeve_count = 0
    
    # Define the output file 
    scraped_file = "scraped_data.json"  # Define scraped_file with the desired filename
    output_file_name = scraped_file
    
    #full path to the input file
    input_file_path = os.path.join(os.getcwd(), "output_links.txt")

    # Check if the input file exists
    if not os.path.exists(input_file_path):
        print("Input file not found:", input_file_path)
        sys.exit()

    # Read the input file and process each line (file path)
    global final_data
    final_data = {}
    with open(input_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            url = line.strip()  # Remove leading/trailing whitespace and newline characters
            get_data(url)

    # Process the data and save it to the output file
    with open(output_file_name, 'w', encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)
    print("File saved:", output_file_name)

# Global variable to count full sleeve products
full_sleeve_count = 0
MAX_THREADS = 30
if __name__ == "__main__":
    main()
