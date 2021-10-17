from attr import attrib, attrs

from qwip._defaults import qwip_defaults
from qwip.settings.settings import Settings, qattrs

@qattrs
class DefaultSettings(Settings):
    def reset(self, *keys):
        """Resets the given parameters to their default values."""
        defaults = default_qsettings()

        for key in keys:
            self.update({key: defaults[key]})

        if len(keys) == 0:
            self.update(defaults)

@qattrs
class DataSettings(Settings):
    base_directory: str
    directory_rule: str
    directory_exist_ok: bool
    date_fmt: str

@qattrs
class QWiPSettings(DefaultSettings):
    """A settings class for global library settings."""
    data: DataSettings

    
def default_qsettings():
    return QWiPSettings(**qwip_defaults)

qsettings = default_qsettings()