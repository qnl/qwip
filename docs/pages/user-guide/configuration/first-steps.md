# First Steps

## Creating a Database

To get started with the configuration database, we need to first create a new database for the device we are working with. Using the `database-setup.py` script we can easily automate the initial setup of creating a new database, creating all the required tables, and making the first commit.

<div class="termy">


```console
$ python .\scripts\database-setup.py create --hostname [host] --username [username]
# Password: $ *****
# Select a name for the new database: $ hero_device
# Repeat for confirmation: $ hero_device
---> 100%
Successfully created database!
Successfully created tables!
---------- hero_device ----------
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Table -------------- ┃ Columns ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ folders ------------ │ 3 ----- │
│ parameters --------- │ 5 ----- │
│ waveforms ---------- │ 5 ----- │
│ waveform_locations - │ 3 ----- │
│ constraints -------- │ 4 ----- │
│ sequence_elements -- │ 2 ----- │
└──────────────────────┴─────────┘
[main talqedlr] Created tables: constraints, folders, parameters, sequence_elements, waveform_locations, waveforms
6 tables added
```

</div>

!!! note
    Configuration databases are similar in scope to git repositories. Each *device* should get its own database, since the system parameters only make sense in the context of a particular device. If multiple users are running different experiments on the same device they can branch the database, which makes it simpler to merge configurations down the line if needed.

## Connecting to the Database

1. **Choose the hostname and the database of the Database.**
``` py title="connect to the database"
def bubble_sort(items):
db = ConfigDB.from_parameters(
    host='[hostname]',
    database='[databasename]',
    schema=ConfigSchema
)

db.connect()

datastore = Datastore(
    storage=HTTPStorageBackend.from_url("http://localhost:4002"),
    url=db.url.set(database="qnl_dataserver")
)
datastore.connect();
```

2. **Login with your credentials.**
```py
Username: 
......
Password: 
······
```

## Configuring the Database
After create and connect to the Database, we need to configure the new Database

1. **All and readout**
    ```python
    db.config.create_all()
    db.config.readout.create_all(default={})
    ```
2. **Setup channels and construct compiler from devices**
    ```python
    from qwip.sequencer.compilation import DeviceInfo, ChannelInfo
    from qwip.backends.qubic import QubicCompiler
    
    qubit = DeviceInfo.from_channels([ChannelInfo(name=f"CH{ch}.qdrv", index=ch, subchannel=0) for ch in range(8)], name="qubit", sample_rate=8e9, dtype=np.complex64)
    readout = DeviceInfo.from_channels([ChannelInfo(name=f"CH{ch}.rdrv", index=ch, subchannel=1) for ch in range(8)], name="readout", sample_rate=0.5e9, dtype=np.complex64)
    adc = DeviceInfo.from_channels([ChannelInfo(name=f"CH{ch}.rdlo", index=ch, read=True, subchannel=2) for ch in range(8)], name="adc", sample_rate=0.5e9, dtype=np.complex64)
    
    compiler = QubicCompiler.from_devices([qubit, readout, adc])
    ```
3. **Pipeline**

    ```python
    from qwip.processing import ReadoutPipeline
    pipeline = ReadoutPipeline()
    ```
4. **Initialize the QPU**

    ```python
    qpu = QPU(db=db, compiler=compiler, pipeline=pipeline, subsystems={}, datastore=datastore)
    qpu.save_compiler()
    ```
5. **Commit all the configuration**
    ```python
    db.config.update(sample_id="[sample_id]", cooldown_id="[cooldown_id]")
    db.commit("Initial setup", add="all")
    ```
