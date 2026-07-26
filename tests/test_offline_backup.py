import hashlib
import tempfile
import unittest
from pathlib import Path

from clinical_app.database import PilotDatabase
from clinical_app.offline_backup import create_backup, verify_backup
from test_database_metrics import case_record


class OfflineBackupTests(unittest.TestCase):
    def test_backup_copies_database_images_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "live"
            image_path = data_dir / "images" / "submission-1" / "image-1.jpg"
            image_path.parent.mkdir(parents=True)
            image_bytes = b"offline-pathology-image-test"
            image_path.write_bytes(image_bytes)
            record = case_record("image-1", predicted=0, accepted=True)
            record.update(
                {
                    "submission_id": "submission-1",
                    "image_storage_key": "submission-1/image-1.jpg",
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "image_content_type": "image/jpeg",
                    "image_size_bytes": len(image_bytes),
                }
            )
            database = PilotDatabase(data_dir / "pilot.db")
            database.insert_submission([record])

            backup_dir = create_backup(data_dir, root / "external-drive", "test")
            manifest = verify_backup(backup_dir)
            self.assertEqual(manifest["patient_case_count"], 1)
            self.assertEqual(manifest["image_record_count"], 1)
            self.assertEqual(manifest["persisted_image_count"], 1)
            self.assertTrue((backup_dir / "images/submission-1/image-1.jpg").is_file())
            self.assertFalse(any(
                item["path"].endswith(("-wal", "-shm"))
                for item in manifest["files"]
            ))
            # Verification must be repeatable and must not create/delete
            # SQLite sidecar files in the completed backup.
            verify_backup(backup_dir)
            verify_backup(backup_dir)


if __name__ == "__main__":
    unittest.main()
