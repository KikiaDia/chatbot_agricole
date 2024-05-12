import requests

url = "http://localhost:8000/api/price_forecast"

data = {
    "regions": "Region1",
    "date": "01/2024",
    "produits": "Riz"
}


response = requests.post(url, json=data)

print(response.json())
