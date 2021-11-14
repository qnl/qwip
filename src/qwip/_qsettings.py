"""QWiP Settings."""
from typing import Optional
from copy import deepcopy
from pathlib import Path

from attr import attrib

from qwip import __version__, __file__
from qwip._repodata import get_repodata
from qwip.settings.settings import Settings, qattrs
from qwip.parameters import Parameters

class DefaultSettings(Settings):
    """A settings class that saves default parameters.

    This settings object allows parameters to be reset to a default parameter.
    """
    __slots__ = ('_defaults',)

    def reset(self, *keys):
        """Resets the given parameters to their default values.
        
        Raises a ValueError if defaults do not exist yet.
        """
        if not hasattr(self, '_defaults'):
            raise ValueError(
                f'Cannot reset {type(self).__name__} because no defaults exist.'
                f' Call set_defaults() first.'
            )

        for key in keys:
            self.update({key: self._defaults[key]})

        if len(keys) == 0:
            self.update(self._defaults)

    def set_defaults(self) -> Parameters:
        """Sets the current state of the settings object as the default.
        
        This method recursively calls set_defaults on all DefaultSettings
        members. Note that set_defaults will change the defaults for all
        instances of DefaultSettings.
        """
        self._defaults = Parameters()
        
        for k, v in self.items():
            if isinstance(v, Settings) and hasattr(v, 'set_defaults'):
                self._defaults[k] = v.set_defaults()
            else:
                self._defaults[k] = deepcopy(v)

        return self._defaults

@qattrs
class DataSettings(DefaultSettings):
    base_directory: str = '.'
    directory_rule: str = 'date'
    directory_exist_ok: bool = True
    date_fmt: str = 'YYYY-MM-DD'

@qattrs
class SlackSettings(DefaultSettings):
    @qattrs
    class SlackChannel(DefaultSettings):
        webhook_url: str

    oauth_url: str = 'https://slack.com/oauth/v2/authorize?client_id=48620956720.2735463917265&scope=incoming-webhook&user_scope='
    channels: Parameters[str, SlackChannel] = attrib(factory=Parameters)

@qattrs
class SourceInfo(DefaultSettings):
    directory: Path = Path(__file__).parent
    repository: Optional[Path] = None
    commit: Optional[str] = None
    branch: Optional[str] = None

@qattrs
class QWiPSettings(DefaultSettings):
    """A settings class for global library settings."""
    version: str = __version__
    src: SourceInfo = attrib(factory=SourceInfo)
    file_format: str = 'yaml'
    notifiers: Parameters[str, SlackSettings] = attrib(factory=Parameters)
    data: DataSettings = attrib(factory=DataSettings)

def process_qsettings_file():
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    paths = ['.', '~/.qwip/']

    for folder in paths:
        path = Path(folder).expanduser().resolve() / 'settings.qwip'
        
        if path.exists() and (p := yaml.load(path)):
            return Parameters(p) #.toflatdict()

    return {}
    
def default_qsettings():
    qsettings = QWiPSettings()
    qsettings.src.update(get_repodata(qsettings.src.directory))

    qsettings['notifiers/slack'] = SlackSettings()

    qsettings.update(process_qsettings_file())
    qsettings = QWiPSettings(**qsettings)

    qsettings.set_defaults()

    return qsettings