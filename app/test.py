import requests
import json
from google.oauth2 import service_account
import google.auth.transport.requests
PROJECT_ID="532895528435"
ENDPOINT_ID="2837575628499189760"

credentials = service_account.Credentials.from_service_account_file('serviceKeys.json', scopes=['https://www.googleapis.com/auth/cloud-platform'])
url = f"https://us-west1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-west1/endpoints/{ENDPOINT_ID}:predict"
auth_req = google.auth.transport.requests.Request()
credentials.refresh(auth_req)
# Create a dictionary to represent the headers
headers = {
   "Authorization": "Bearer " + credentials.token,
    "Content-Type": "application/json",  # Include other headers as needed
}
data = {
  "instances":[
{"P10": "38",
      "P15": "40",
      "P25": "42",
      "PMAX": "45",
      "PCOUNT": "62",
      "PAT": "0",
      "PATCON": "-67.66",
      "UNR": "6.5",
  "UNRCON": "2.48"}
]
}

json_data = json.dumps(data)
response = requests.post(url, headers=headers,data = json_data)

if response.status_code == 200:
    data = response.json()
else:
    print(f"Request failed with status code: {response.status_code}")
print(data)
