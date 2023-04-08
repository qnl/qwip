from functools import lru_cache

from loguru import logger

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import AnyHttpUrl, BaseSettings


class Settings(BaseSettings):
    SLACK_CLIENT_ID: str = "client_id"
    SLACK_CLIENT_SECRET: str = "secret"
    SLACK_REDIRECT_URI: AnyHttpUrl = "https://localhost:8000/QWiP/app/"
    SLACK_OAUTH_URL: AnyHttpUrl = "https://slack.com/api/oauth.v2.access"
    DOCS_URL: AnyHttpUrl = "http://localhost:8080/QWiP/user-guide/workflow/"
    BASE_PATH: str = "/"

    class Config:
        env_file = ".env"


router = APIRouter()


@lru_cache()
def get_settings():
    return Settings()


async def oauth_token_exchange(
    code: str, state: str, settings: Settings
) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.SLACK_OAUTH_URL,
            data=dict(
                code=code,
                client_id=settings.SLACK_CLIENT_ID,
                client_secret=settings.SLACK_CLIENT_SECRET,
                redirect_uri=settings.SLACK_REDIRECT_URI,
            ),
        )

    return response


async def process_response(data: dict, settings: Settings) -> RedirectResponse:
    if not data["ok"]:
        error = data["error"]
        return RedirectResponse(
            f"{settings.DOCS_URL}?webhook_url={error}#adding-a-webhook"
        )

    webhook_url = data["incoming_webhook"]["url"]
    channel = data["incoming_webhook"]["channel"]

    return RedirectResponse(
        f"{settings.DOCS_URL}?webhook_url={webhook_url}&channel={channel}#adding-a-webhook"
    )


def post_welcome_message(webhook_url: str, settings: Settings):
    text = (
        f"Your new webhook url is {webhook_url}\n"
        f"Add this to your `settings.qwip` file to send messages here from QWiP. "
        f"See the <{settings.DOCS_URL}|docs> for more details."
    )

    httpx.post(
        webhook_url,
        json={
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            ]
        },
    )


@router.get("/register")
async def slack_redirect(
    request: Request,
    background_tasks: BackgroundTasks,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    if error:
        return RedirectResponse(f"{settings.DOCS_URL}?error={error}#adding-a-webhook")

    response = await oauth_token_exchange(code, state, settings)
    data = response.json()

    redirect = await process_response(data, settings)
    background_tasks.add_task(
        post_welcome_message, data["incoming_webhook"]["url"], settings
    )

    return redirect


settings = Settings()

logger.info(f"Using the following environment settings: {settings}")

app = FastAPI(
    title="QWiP Web App",
)


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
