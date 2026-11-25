import os
import subprocess
from pathlib import Path


def test_failed_encryption_does_not_leave_plaintext_backup(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    tools = {
        "pg_dump": 'test "$1" = postgresql://localhost/phosca || exit 42\necho fixture_data',
        "rsync": 'cp -r "$3" "$4"',
        "gpg": 'test -s "$6" || exit 43\necho partial > "$5"\nexit 61',
    }
    for name, code in tools.items():
        command = commands / name
        command.write_text(f"#!/bin/sh\n{code}\n")
        command.chmod(0o700)
    storage = tmp_path / "source"
    storage.mkdir()
    (storage / "original.txt").write_text("fixture")
    destination = tmp_path / "backups"
    script = Path(__file__).resolve().parents[2] / "scripts" / "backup.sh"
    result = subprocess.run(
        ["bash", script],
        env={
            **os.environ,
            "PATH": os.pathsep.join((str(commands), os.environ["PATH"])),
            "DATABASE_URL": "postgresql+asyncpg://localhost/phosca",
            "STORAGE_PATH": str(storage),
            "BACKUP_ROOT": str(destination),
            "GPG_RECIPIENT": "isolated test key",
        },
        capture_output=True,
    )
    assert result.returncode == 61, result.stderr.decode()
    assert list(destination.iterdir()) == []
