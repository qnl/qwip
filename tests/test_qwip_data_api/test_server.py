import pytest

from fastapi import status
from fastapi.testclient import TestClient
from qwip_data_api.server import app
from pendulum import DateTime


client = TestClient(app, base_url = "http://localhost:8000")

base_url = "api/v1/"

@pytest.mark.parametrize(
    "exact_criteria", 
    [
        [],
        [
            ("sample_id", "K99999"), 
            ("time_start", "2024-05-28T00:30:47-07:00")
        ],
        [
            ("sample_id", "K99999"),
            ("cooldown_id", "SNB240514"),
        ]
    ]
)
def test_get_datasets(client, exact_criteria):
    s = ""
    for c, v in exact_criteria:
        s = s + "&" + c + "=" +  v

    url = base_url + "datasets/?" + s
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    if exact_criteria:
        for elem in response.json():
            assert [elem[c] == v for c, v in exact_criteria]

@pytest.mark.parametrize(
    "IDs",
    [
        ["018fbe19-edcb-7d6e-a689-078466b7a1a1"], # non-matching UUID, how to test? 
        ["018fbe0c-dfb0-7122-b4a7-fcff9b01d485", "018fbe1a-001c-7dd5-9734-d6c7adaf3c34"] # matching UUIDs
    ] 
)
def test_get_dataset_by_id(client, IDs):
    for id in IDs:
        url = base_url + "datasets/" + id
        response = client.get(url)
        if response.json():
            assert response.json()['id'] == id