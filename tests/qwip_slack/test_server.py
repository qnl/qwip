import httpx
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from qwip_slack.server import app, settings

client = TestClient(app)

success = {
    "ok": True,
    "incoming_webhook": {
        "channel": "@qwip",
        "url": "https://hooks.slack.com/services/team_id/hook",
    },
}


def test_register_success(respx_mock):
    webhook_url = success["incoming_webhook"]["url"]
    channel = success["incoming_webhook"]["channel"]

    oauth = respx_mock.post(settings.SLACK_OAUTH_URL).mock(
        return_value=httpx.Response(200, json=success)
    )
    webhook = respx_mock.post(webhook_url).mock(return_value=httpx.Response(200))

    response = client.get("/register?code=1234", follow_redirects=False)

    assert len(oauth.calls) == 1
    assert len(webhook.calls) == 1
    assert response.status_code == status.HTTP_307_TEMPORARY_REDIRECT
    assert (
        response.headers["location"]
        == settings.DOCS_URL
        + f"?webhook_url={webhook_url}&channel={channel}#adding-a-webhook"
    )


def test_register_error(respx_mock):
    response = client.get("/register?error=error", follow_redirects=False)

    assert response.status_code == status.HTTP_307_TEMPORARY_REDIRECT
    assert (
        response.headers["location"]
        == settings.DOCS_URL + "?error=error#adding-a-webhook"
    )
