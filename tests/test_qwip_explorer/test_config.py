from qwip_explorer.config import ExplorerDefaults


def test_generic_defaults_do_not_embed_a_personal_connection(monkeypatch):
    for name in (
        "QWIP_DB_HOST", "QWIP_DATASTORE_DB", "QWIP_CONFIG_DB", "QWIP_DB_USERNAME",
        "QWIP_DB_PASSWORD", "QWIP_STORAGE_URL", "QWIP_LOCAL_DATASTORE",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = ExplorerDefaults.from_environment()

    assert defaults.host == ""
    assert defaults.datastore_database == ""
    assert defaults.configuration_database == ""
    assert defaults.username == ""
    assert defaults.local_storage_directory == ""
    assert defaults.storage_mode == "HTTP"
