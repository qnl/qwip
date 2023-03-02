import traceback
from contextlib import contextmanager
from functools import partial

from IPython.core.getipython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class, register_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from notifiers import get_notifier

from qwip import qsettings
from qwip.defaults import QWiPDefault


@contextmanager
def slack_notify(channel, message="Finished!"):
    provider = get_notifier("slack")

    if channel not in qsettings["slack/channels"]:
        raise KeyError(f"'{channel}' is not a configured Slack channel")

    kwargs = qsettings[f"slack/channels/{channel}"]

    notify = partial(provider.notify, **kwargs)
    failed = False
    try:
        yield notify
    except Exception as e:
        notify(message=f"*{type(e).__name__}*: {e}", **kwargs)
        failed = True
        raise e
    finally:
        if not failed:
            notify(message=message, **kwargs)


@magics_class
class SlackNotifyMagic(Magics):
    @magic_arguments()
    @argument("channel", help="Channel to send the notification to.")
    @argument("-m", "--message", default="Finished!", help="Message to send.")
    @cell_magic
    def slack_notify(self, line, cell):
        args = parse_argstring(self.slack_notify, line)

        provider = get_notifier("slack")
        channel = args.channel
        if channel not in qsettings["slack/channels"]:
            raise KeyError(f"'{channel}' is not a configured Slack channel.")

        slack_kwargs = qsettings[f"slack/channels/{channel}"]

        output = get_ipython().run_cell(cell)

        if output.success:
            provider.notify(message=args.message, **slack_kwargs)
        else:
            tb = "".join(traceback.format_exception(output.error_in_exec))
            message = f"```\n" f"{tb}\n" f"```"
            provider.notify(message=message, **slack_kwargs)


def load_ipython_extension(ipython):
    ipython.register_magics(SlackNotifyMagic)
