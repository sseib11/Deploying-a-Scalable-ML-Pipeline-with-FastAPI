import requests

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    # GET
    r = requests.get(f"{BASE_URL}/", timeout=10)
    print(f"Status Code: {r.status_code}")
    print(f"Result: {r.json()}")

    # POST
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    print(f"Status Code: {r.status_code}")
    print(f"Result: {r.json()}")


if __name__ == "__main__":
    main()
