# First Steps

To get started with the configuration database, we need to first create a new database for the device we are working with.

!!! note

    Configuration databases are similar in scope to git repositories. Each *device* should get its own database, since the system parameters only make sense in the context of a particular devices. If multiple users are running different experiments on the same device they can branch the database, which makes it simpler to merge configurations down the line if needed.

