import httpx
from fastapi.testclient import TestClient


def test_get_status(client: TestClient, auth: httpx.BasicAuth) -> None:
    response = client.get("/api/dolt/test_db/status", auth=auth)

    assert response.status_code == 200
    assert response.json() == []


def test_log(client: TestClient, auth: httpx.BasicAuth) -> None:
    response = client.get("/api/dolt/test_db/log", auth=auth)

    assert response.status_code == 200
    assert response.json() == [{
        "hash": "0c4nibig1790espsukdmk2tedu73b3h6",
        "committer": "Dolt System Account",
        "email": "doltuser@dolthub.com",
        "date": "2023-02-01T07:46:02.107000+00:00",
        "message": "Initialize data repository",
    }]
