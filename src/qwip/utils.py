import inspect
import traceback
import warnings
from contextlib import contextmanager
from functools import partial, wraps

from IPython.core.getipython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class, register_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from notifiers import get_notifier

from qwip import qsettings


def deprecated(*, version: str, removed: str, message: str = ""):
    """Decorator for deprecated functions, methods, and classes.

    This ensures that a DeprecationWarning is thrown when the specified target is called
    by a user.

    Args:
        version: The version where the target was first deprecated.
        removed: The version where the target will be removed.
        message: A message for the user regarding the reason for the deprecation or an
            alternative to use instead.

    Returns:
        The wrapped function, method, or class.
    """

    def decorator(target):
        @wraps(target)
        def wrapper(*args, **kwargs):
            name = f"{target.__module__}.{target.__qualname__}"

            warnings.warn(
                (
                    f"`{name}` is deprecated since {version} and will be removed in "
                    f"{removed}.\n{message}"
                ),
                stacklevel=2,
                category=DeprecationWarning,
            )

            return target(*args, **kwargs)

        return wrapper

    return decorator


def format_exception(exc: Exception) -> str:
    """Formats an exception traceback as a string.

    Args:
        exc: An `Exception` object.

    Returns:
        A string with the exception traceback.
    """

    tb = "".join(traceback.format_exception(exc))
    return f"```\n" f"{tb}\n" f"```"


@contextmanager
def slack_notify(channel: str, message: str = "Finished!"):
    """Context manager for sending a slack notification upon completion.

    Args:
        channel: The name of the channel. This should correspond to a registered channel
            in `qsettings["slack/channels"]`
        message: The message to send upon successful completion
    """
    provider = get_notifier("slack")

    if channel not in qsettings["slack/channels"]:
        raise KeyError(f"'{channel}' is not a configured Slack channel")

    kwargs = qsettings[f"slack/channels/{channel}"]

    notify = partial(provider.notify, **kwargs)

    try:
        yield notify
    except Exception as e:
        message = format_exception(e)
        raise e
    finally:
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
            provider.notify(
                message=format_exception(output.error_in_exec), **slack_kwargs
            )


def load_ipython_extension(ipython):
    ipython.register_magics(SlackNotifyMagic)
