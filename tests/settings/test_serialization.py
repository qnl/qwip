from attrs import field

from qwip.settings.settings import Settings, qdefine
from qwip.flatdict import FlatDict


def test_flatdict_structure():
    @qdefine
    class OuterSettings(Settings):
        @qdefine
        class InnerSettings(Settings):
            a: str

        children: FlatDict[str, InnerSettings] = field(factory=FlatDict)

    settings = OuterSettings(children=FlatDict(c1=dict(a=1), c2=dict(a=2)))
    assert settings == OuterSettings(
        children=FlatDict(
            c1=OuterSettings.InnerSettings(a='1'),
            c2=OuterSettings.InnerSettings(a='2')
        )
    )