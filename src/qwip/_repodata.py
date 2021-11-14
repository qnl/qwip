import subprocess

from pathlib import Path

def get_repodata(directory: str) -> dict:
    """Gets data about the current state of the repository.

    Args:
        directory (str): The path to the git repository.
    
    Returns:
        (dict): A dictionary with information about the repository.
    """

    cmds = {
        'repository': ['git', 'rev-parse', '--show-toplevel'],
        'commit': ['git', 'rev-parse', 'HEAD'],
        'branch': ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
    }

    if not isinstance(directory, Path):
        directory = Path(directory)

    data = dict(directory=directory.expanduser().resolve())

    for key, cmd in cmds.items():
        process = subprocess.run(
            cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if not process.returncode:
            data[key] = process.stdout.strip().decode('utf-8')
    
    if data['repository']:
        data['repository'] = Path(data['repository'])

    return data

