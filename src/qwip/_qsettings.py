from qwip._defaults import qwip_defaults
from qwip.settings.settings import Settings, qwip_attrs

@qwip_attrs
class QWiPSettings(Settings):
    """A settings class for global library settings."""

    def reset(self, *keys):
        """Resets the given parameters to their default values."""
        defaults = default_qsettings()

        for key in keys:
            self.update({key: defaults[key]})

        if len(keys) == 0:
            self.update(defaults)

    
def default_qsettings():
    return QWiPSettings(**qwip_defaults)

qsettings = default_qsettings()