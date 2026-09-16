# Datastore Explorer

The optional QWIP Datastore Explorer is a local, read-only browser for QWIP dataset
catalogs, stored assets, and Dolt configuration history. Each collaborator starts it
on their own computer and connects with their own database and asset-storage access.
It is currently available on the `codex/datastore-explorer` branch and is not yet part
of a published QWIP release.

## Install and launch

Use Python 3.12 or newer. For a checkout of this repository, install QWIP and the
optional Explorer dependencies in editable mode:

```bash
python -m pip install -e ".[explorer]"
```

After a QWIP release includes this feature, install the same optional dependency group:

```bash
python -m pip install "qwip[explorer]"
```

Start a localhost-only server:

```bash
qwip-explorer
```

It opens at `http://127.0.0.1:8501`. Use another unused port when needed:

```bash
qwip-explorer --port 8600
# or: QWIP_EXPLORER_PORT=8600 qwip-explorer
```

`python -m qwip_explorer` is equivalent. The launcher finds the installed application,
so it works from any current directory. It binds to `127.0.0.1`, keeping the Explorer
local to the computer that started it.

## Configure a connection

Enter the database host, datastore database, optional configuration database, username,
password, and asset storage location in the sidebar. The form starts blank rather than
using a lab-specific server, database, account, or drive path. Passwords are masked.

You can prefill the same fields with environment variables:

| Variable | Meaning |
| --- | --- |
| `QWIP_DB_HOST` | Dolt/MySQL-compatible database host |
| `QWIP_DATASTORE_DB` | Datastore metadata database |
| `QWIP_CONFIG_DB` | Optional configuration database for commit history |
| `QWIP_DB_USERNAME` | Database username |
| `QWIP_DB_PASSWORD` | Database password |
| `QWIP_STORAGE_MODE` | `HTTP` or `Local folder` |
| `QWIP_STORAGE_URL` | HTTP asset-server URL |
| `QWIP_LOCAL_DATASTORE` | Local asset root when using `Local folder` |
| `QWIP_TIMEZONE` | Dataset-search timezone; defaults to `America/Los_Angeles` |
| `QWIP_EXPLORER_PORT` | Launcher port; defaults to `8501` |

Access to the database and stored assets usually requires the organization network or
VPN, plus permission for the chosen database and asset backend. Installing the UI does
not grant that access.

## Troubleshooting

- If `qwip-explorer` is unavailable, use `python -m qwip_explorer` from the same
  environment where QWIP was installed.
- If port 8501 is already in use, choose another port with `qwip-explorer --port 8600`.
- On Windows, socket error `10013` can mean the launching environment blocks network
  access. Launch the Explorer from your normal terminal in the installed Python
  environment, with access to the lab network and any mapped storage drives. A
  server started inside a restricted development sandbox may display the interface
  successfully while being unable to reach the database or measurement files.
- Connection or asset-load errors normally indicate that the computer lacks the needed
  network/VPN route, database credentials, or asset-storage access. Confirm those
  details with the data owner before changing Explorer settings.

## Update

For a released QWIP version, upgrade QWIP and its optional Explorer dependencies together:

```bash
python -m pip install --upgrade "qwip[explorer]"
```

For an editable checkout, update the checkout and rerun the editable install command
from the first section.

The Explorer retains no database credentials in its package defaults. Each browser
session owns its database connections; reconnecting closes its previous connection.

## Measurement recipes

Reusable measurement metadata helpers are available without installing Streamlit:

```python
from qwip.data import run_with_recipe
```

`run_with_recipe` stores a versioned recipe, fixed parameters, sweep parameters, and
the executed sequence with a QPU run. The Explorer displays these assets when present.
