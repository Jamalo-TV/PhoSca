#!/usr/bin/env bash
set -euo pipefail
umask 077

BACKUP_ROOT="${BACKUP_ROOT:-/backups/album-digitizer}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATABASE_URL="${DATABASE_URL:?DATABASE_URL is required}"
DATABASE_URL="${DATABASE_URL/#postgresql+asyncpg:/postgresql:}"
STORAGE_PATH="${STORAGE_PATH:-/app/storage}"
GPG_RECIPIENT="${GPG_RECIPIENT:-}"

mkdir -p -- "${BACKUP_ROOT}"
WORKDIR="$(mktemp -d -- "${BACKUP_ROOT}/.phosca-backup.XXXXXX")"
trap 'rm -rf -- "${WORKDIR}"' EXIT
SNAPSHOT="${WORKDIR}/${TIMESTAMP}"
mkdir -- "${SNAPSHOT}"
pg_dump "${DATABASE_URL}" > "${SNAPSHOT}/database.sql"
rsync -a --delete "${STORAGE_PATH}/" "${SNAPSHOT}/storage/"

ARCHIVE="${WORKDIR}/${TIMESTAMP}.tar.gz"
ENCRYPTED_ARCHIVE="${ARCHIVE}.gpg"
tar -C "${WORKDIR}" -czf "${ARCHIVE}" "${TIMESTAMP}"
if [[ -n "${GPG_RECIPIENT}" ]]; then
  gpg --encrypt --recipient "${GPG_RECIPIENT}" --output "${ENCRYPTED_ARCHIVE}" "${ARCHIVE}"
else
  gpg --symmetric --cipher-algo AES256 --output "${ENCRYPTED_ARCHIVE}" "${ARCHIVE}"
fi

ARCHIVE_SHA256="$(sha256sum "${ENCRYPTED_ARCHIVE}" | awk '{print $1}')"
ARCHIVE_BYTES="$(stat -c '%s' "${ENCRYPTED_ARCHIVE}")"
printf '{"schema_version":1,"created_at":"%s","archive":"%s","sha256":"%s","bytes":%s,"restore_tested":false}\n' \
  "${TIMESTAMP}" "$(basename "${ENCRYPTED_ARCHIVE}")" "${ARCHIVE_SHA256}" "${ARCHIVE_BYTES}" > "${ENCRYPTED_ARCHIVE}.evidence.json"
DESTINATION="${BACKUP_ROOT}/${TIMESTAMP}.tar.gz.gpg"
mv -nT -- "${ENCRYPTED_ARCHIVE}" "${DESTINATION}"
if [[ -e "${ENCRYPTED_ARCHIVE}" ]]; then
  echo "Backup already exists at ${DESTINATION}" >&2
  exit 1
fi
if ! mv -nT -- "${ENCRYPTED_ARCHIVE}.evidence.json" "${DESTINATION}.evidence.json" ||
    [[ -e "${ENCRYPTED_ARCHIVE}.evidence.json" ]]; then
  rm -f -- "${DESTINATION}"
  exit 1
fi
echo "Encrypted backup and evidence written under ${BACKUP_ROOT}"
