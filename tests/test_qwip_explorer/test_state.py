from qwip_explorer.state import clear_connection_results


def test_reconnect_clears_all_prior_connection_results():
    state = {
        "connected_username": "old-user",
        "connected_datastore_database": "old-catalog",
        "connected_configuration_database": "old-config",
        "dataset_records": [{"id": "old"}],
        "structure_branch_commits": {"branch": "old"},
        "unrelated": "retain",
    }

    clear_connection_results(state)

    assert state == {"unrelated": "retain"}
