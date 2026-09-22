import requests
from bs4 import BeautifulSoup

def scrape_sws_solar_data():
    url = "https://www.sws.bom.gov.au/Solar"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract Solar Flux (SFI)
    sfi_element = soup.find("td", string="Solar Flux (10.7cm)")
    sfi = sfi_element.find_next("td").text.strip() if sfi_element else "N/A"

    # Extract K-Index
    k_index_url = "https://www.sws.bom.gov.au/Geophysical"
    k_response = requests.get(k_index_url)
    k_soup = BeautifulSoup(k_response.text, "html.parser")
    k_index_element = k_soup.find("td", string="Estimated K-index")
    k_index = k_index_element.find_next("td").text.strip() if k_index_element else "N/A"

    return {
        "solar_flux": sfi,
        "k_index": k_index,
    }

# Example usage:
data = scrape_sws_solar_data()
print(f"Solar Flux: {data['solar_flux']}, K-Index: {data['k_index']}")