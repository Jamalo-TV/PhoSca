# Operations

## Backup

Run `scripts/backup.sh` from the backend or a maintenance container with `pg_dump`, `rsync`, and `gpg` available.

Required environment:

- `DATABASE_URL`: PostgreSQL connection string.
- `STORAGE_PATH`: storage volume path, default `/app/storage`.
- `BACKUP_ROOT`: external drive or NAS mount, default `/backups/album-digitizer`.

Optional:

- `GPG_RECIPIENT`: public-key recipient. If omitted, the script uses symmetric AES-256 encryption.

## Restore

1. Decrypt the backup archive with `gpg`.
2. Stop backend and worker containers.
3. Restore PostgreSQL with `psql "$DATABASE_URL" < database.sql`.
4. Sync `storage/` back into the configured storage volume.
5. Start services and run a small album retrieval smoke test.

## Future Rust Extraction Points

Only consider Rust if profiling shows Python bottlenecks after load testing. Candidate boundaries are contour extraction, pHash batch computation, and EXIF batch rewrites. Keep the Python API contract unchanged.

