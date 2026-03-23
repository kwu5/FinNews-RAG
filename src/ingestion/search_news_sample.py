import requests


def my_custom_function():
    url = "https://api.worldnewsapi.com/search-news?text=stock&language=en&earliest-publish-date=2026-03-22"
    api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    headers = {
        'x-api-key': api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    print(my_custom_function())
