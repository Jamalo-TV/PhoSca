#!/usr/bin/env bash
set -euo pipefail

BACKUP_ROOT="${BACKUP_ROOT:-/backups/album-digitizer}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR="${BACKUP_ROOT}/${TIMESTAMP}"
DATABASE_URL="${DATABASE_URL:?DATABASE_URL is required}"
STORAGE_PATH="${STORAGE_PATH:-/app/storage}"
GPG_RECIPIENT="${GPG_RECIPIENT:-}"

mkdir -p "${WORKDIR}"
pg_dump "${DATABASE_URL}" > "${WORKDIR}/database.sql"
rsync -a --delete "${STORAGE_PATH}/" "${WORKDIR}/storage/"

tar -C "${BACKUP_ROOT}" -czf "${WORKDIR}.tar.gz" "${TIMESTAMP}"
if [[ -n "${GPG_RECIPIENT}" ]]; then
  gpg --yes --encrypt --recipient "${GPG_RECIPIENT}" "${WORKDIR}.tar.gz"
  rm -f "${WORKDIR}.tar.gz"
else
  gpg --yes --symmetric --cipher-algo AES256 "${WORKDIR}.tar.gz"
  rm -f "${WORKDIR}.tar.gz"
fi

rm -rf "${WORKDIR}"
ENCRYPTED_ARCHIVE="${WORKDIR}.tar.gz.gpg"
ARCHIVE_SHA256="$(sha256sum "${ENCRYPTED_ARCHIVE}" | awk '{print $1}')"
ARCHIVE_BYTES="$(stat -c '%s' "${ENCRYPTED_ARCHIVE}")"
printf '{"schema_version":1,"created_at":"%s","archive":"%s","sha256":"%s","bytes":%s,"restore_tested":false}\n' \
  "${TIMESTAMP}" "$(basename "${ENCRYPTED_ARCHIVE}")" "${ARCHIVE_SHA256}" "${ARCHIVE_BYTES}" > "${ENCRYPTED_ARCHIVE}.evidence.json"
echo "Encrypted backup and evidence written under ${BACKUP_ROOT}"
