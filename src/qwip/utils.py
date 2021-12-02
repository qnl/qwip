from contextlib import contextmanager
from functools import partial

from notifiers import get_notifier

from qwip import qsettings
from qwip.defaults import QWiPDefault

@contextmanager
def slack_notify(channel, message='Finished!'):
    provider = get_notifier('slack')

    if channel not in qsettings['notifiers/slack/channels']:
        raise KeyError(f"'{channel}' is not a configured Slack channel")

    kwargs = qsettings[f'notifiers/slack/channels/{channel}']

    notify = partial(provider.notify, **kwargs)
    failed = False
    try:
        yield notify
    except Exception as e:
        notify(message=f'*{type(e).__name__}*: {e}', **kwargs)
        failed = True
        raise e
    finally:
        if not failed:
            notify(message=message, **kwargs)