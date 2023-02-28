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

To connect to the database.


