"""Database layer: driver, migrations, security and repositories.

These tests run against a throwaway SQLite store so the whole data layer is
exercised without a MySQL server. The MySQL-specific translation is tested
separately by inspecting the generated statements.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from visionai.database.driver import (
    Database,
    DatabaseConfig,
    DatabaseError,
    ddl_suffix,
    placeholder,
    temporary_database,
)
from visionai.database.migrator import (
    Migrator,
    apply_all,
    discover,
    require_up_to_date,
    split_statements,
    to_mysql,
    to_sqlite,
    translate,
)
from visionai.database.repositories import open_repositories
from visionai.security.passwords import (
    Permission,
    hash_password,
    needs_rehash,
    role_has_permission,
    verify_password,
)

STRONG_PASSWORD = "Correct-Horse-Battery-2026"


@pytest.fixture
def database() -> Iterator[Database]:
    db = Database(DatabaseConfig(sqlite_path=":memory:"))
    yield db
    db.close()


@pytest.fixture
def migrated(database: Database) -> Database:
    apply_all(database)
    return database


@pytest.fixture
def repos(migrated: Database):
    return open_repositories(migrated)


@pytest.fixture
def admin(repos):
    return repos.users.create("admin", "admin@example.com", STRONG_PASSWORD, role="administrator")


class TestDriver:
    def test_sqlite_round_trip(self, database: Database) -> None:
        database.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)")
        identifier = database.insert("INSERT INTO demo (name) VALUES (%s)", ("a",))
        assert identifier == 1
        assert database.query("SELECT * FROM demo") == [{"id": 1, "name": "a"}]

    def test_query_one_returns_none_when_absent(self, database: Database) -> None:
        assert database.query_one("SELECT 1 AS x WHERE 0") is None

    def test_scalar_and_default(self, database: Database) -> None:
        assert database.scalar("SELECT 7 AS x") == 7
        assert database.scalar("SELECT NULL AS x", default=3) == 3

    def test_executemany(self, database: Database) -> None:
        database.execute("CREATE TABLE t (v TEXT)")
        assert database.executemany("INSERT INTO t (v) VALUES (%s)", [("a",), ("b",)]) == 2
        assert database.scalar("SELECT COUNT(*) AS n FROM t") == 2

    def test_executemany_with_no_rows(self, database: Database) -> None:
        database.execute("CREATE TABLE t (v TEXT)")
        assert database.executemany("INSERT INTO t (v) VALUES (%s)", []) == 0

    def test_bad_sql_raises_database_error(self, database: Database) -> None:
        with pytest.raises(DatabaseError):
            database.query("SELECT * FROM does_not_exist")

    def test_transaction_rolls_back(self, database: Database) -> None:
        database.execute("CREATE TABLE t (v TEXT)")
        with pytest.raises(RuntimeError):
            with database.transaction():
                database.execute("INSERT INTO t (v) VALUES (%s)", ("x",))
                raise RuntimeError("boom")
        assert database.scalar("SELECT COUNT(*) AS n FROM t") == 0

    def test_transaction_commits(self, database: Database) -> None:
        database.execute("CREATE TABLE t (v TEXT)")
        with database.transaction():
            database.execute("INSERT INTO t (v) VALUES (%s)", ("x",))
        assert database.scalar("SELECT COUNT(*) AS n FROM t") == 1

    def test_ping_and_describe(self, migrated: Database) -> None:
        assert migrated.ping() is True
        info = migrated.describe()
        assert info["connected"] is True
        assert info["tables"] > 20

    def test_closed_database_refuses_work(self, database: Database) -> None:
        database.close()
        with pytest.raises(DatabaseError):
            database.query("SELECT 1 AS x")

    def test_config_never_exposes_the_password(self) -> None:
        config = DatabaseConfig(password="hunter2-super-secret")
        assert "hunter2" not in json.dumps(config.safe())
        assert config.safe()["password"] == "***"

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MYSQL_HOST", "db.internal")
        monkeypatch.setenv("MYSQL_PORT", "3307")
        monkeypatch.setenv("MYSQL_USER", "app")
        monkeypatch.setenv("MYSQL_PASSWORD", "secret")
        config = DatabaseConfig.from_env()
        assert config.host == "db.internal"
        assert config.port == 3307
        assert config.password == "secret"

    def test_sqlite_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISIONAI_SQLITE_PATH", "/tmp/x.db")
        assert DatabaseConfig.from_env().sqlite_path == "/tmp/x.db"

    def test_invalid_port_is_rejected(self) -> None:
        from visionai.ai.errors import ConfigError

        with pytest.raises(ConfigError):
            DatabaseConfig(port=99999).validate()

    def test_temporary_database_helper(self, tmp_path: Path) -> None:
        db = temporary_database(tmp_path)
        assert db.query("SELECT 1 AS x")[0]["x"] == 1
        db.close()

    @pytest.mark.parametrize(
        ("dialect", "expected"), [("mysql", "%s"), ("sqlite", "?")]
    )
    def test_placeholder(self, dialect: str, expected: str) -> None:
        assert placeholder(dialect) == expected

    def test_ddl_suffix_differs_by_dialect(self) -> None:
        assert "InnoDB" in ddl_suffix("mysql")
        assert ddl_suffix("sqlite") == ""


class TestMigrations:
    def test_all_seven_are_discovered(self) -> None:
        labels = [m.label for m in discover()]
        assert labels == [
            "0001_core", "0002_taxonomy", "0003_datasets", "0004_models",
            "0005_experiments", "0006_sessions", "0007_operations",
        ]

    def test_versions_are_unique_and_ordered(self) -> None:
        versions = [m.version for m in discover()]
        assert versions == sorted(versions)
        assert len(versions) == len(set(versions))

    def test_expected_tables_exist(self, migrated: Database) -> None:
        tables = set(migrated.table_names())
        for expected in (
            "users", "roles", "permissions", "role_permissions", "user_profiles", "settings",
            "emotion_classes", "camera_devices", "datasets", "dataset_images", "dataset_labels",
            "models", "model_versions", "training_runs", "training_metrics", "test_runs",
            "test_results", "confusion_matrix", "experiments", "experiment_metrics",
            "detection_sessions", "face_detections", "expression_results",
            "application_logs", "system_metrics", "hardware_logs",
        ):
            assert expected in tables, f"missing table {expected}"

    def test_migration_is_recorded(self, migrated: Database) -> None:
        rows = migrated.query("SELECT version, name, checksum FROM schema_migrations ORDER BY version")
        assert len(rows) == 7
        assert all(row["checksum"] for row in rows)

    def test_migrating_twice_is_a_no_op(self, database: Database) -> None:
        apply_all(database)
        assert Migrator(database).migrate() == []

    def test_status_reports_up_to_date(self, migrated: Database) -> None:
        assert Migrator(migrated).status().is_up_to_date is True

    def test_status_lists_pending(self, database: Database) -> None:
        status = Migrator(database).status()
        assert len(status.pending) == 7
        assert status.is_up_to_date is False

    def test_dry_run_changes_nothing(self, database: Database) -> None:
        assert len(Migrator(database).migrate(dry_run=True)) == 7
        assert Migrator(database).status().pending

    def test_target_stops_early(self, database: Database) -> None:
        assert Migrator(database).migrate(target=1) == ["0001_core"]
        assert len(Migrator(database).status().pending) == 6

    def test_require_up_to_date_raises_when_behind(self, database: Database) -> None:
        with pytest.raises(DatabaseError):
            require_up_to_date(database)

    def test_require_up_to_date_passes_after_migrating(self, migrated: Database) -> None:
        require_up_to_date(migrated)

    def test_checksum_drift_is_detected(self, migrated: Database, tmp_path: Path) -> None:
        source = next(m for m in discover() if m.version == 1)
        copy_dir = tmp_path / "migrations"
        copy_dir.mkdir()
        for migration in discover():
            (copy_dir / migration.path.name).write_text(migration.read(), encoding="utf-8")
        (copy_dir / source.path.name).write_text(
            source.read() + "\n-- an edit after the fact\n", encoding="utf-8"
        )
        status = Migrator(migrated, copy_dir).status()
        assert status.drifted
        assert "0001_core" in status.drifted[0]

    def test_comments_are_stripped(self) -> None:
        statements = split_statements("-- a comment\nCREATE TABLE t (v TEXT);\n\n")
        assert statements == ["CREATE TABLE t (v TEXT)"]

    def test_multiple_statements_per_file(self) -> None:
        statements = next(m for m in discover() if m.version == 1).statements()
        assert len(statements) >= 5

    def test_unknown_filename_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        assert discover(tmp_path) == []

    def test_rollback_removes_the_last_record(self, migrated: Database) -> None:
        assert Migrator(migrated).rollback_last() == "operations"
        assert len(Migrator(migrated).status().pending) == 1

    def test_describe(self, migrated: Database) -> None:
        info = Migrator(migrated).describe()
        assert info["total"] == 7
        assert info["applied"] == 7
        assert info["up_to_date"] is True


class TestDialectTranslation:
    def test_mysql_gets_autoincrement(self) -> None:
        assert "AUTO_INCREMENT" in to_mysql("id INTEGER PRIMARY KEY AUTOINCREMENT")

    def test_sqlite_keeps_autoincrement(self) -> None:
        assert "AUTOINCREMENT" in to_sqlite("id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY")

    def test_mysql_gets_smallint_as_tinyint(self) -> None:
        assert "TINYINT" in to_mysql("flag SMALLINT NOT NULL DEFAULT 0")

    def test_sqlite_keeps_smallint(self) -> None:
        assert "SMALLINT" in to_sqlite("flag TINYINT NOT NULL DEFAULT 0")

    def test_mysql_create_table_gains_table_options(self) -> None:
        statement = to_mysql("CREATE TABLE roles (\n id INTEGER PRIMARY KEY AUTOINCREMENT\n)")
        assert "ENGINE=InnoDB" in statement
        assert "utf8mb4" in statement

    def test_sqlite_create_table_has_no_table_options(self) -> None:
        assert "ENGINE" not in to_sqlite("CREATE TABLE roles (\n id INTEGER PRIMARY KEY AUTOINCREMENT\n)")

    def test_indexes_are_not_given_table_options(self) -> None:
        assert "ENGINE" not in to_mysql("CREATE INDEX ix_users_role ON users (role_id)")

    def test_translate_dispatches(self) -> None:
        assert "AUTO_INCREMENT" in translate(
            "id INTEGER PRIMARY KEY AUTOINCREMENT", "mysql"
        )
        assert "AUTOINCREMENT" in translate(
            "id INTEGER PRIMARY KEY AUTOINCREMENT", "sqlite"
        )

    def test_every_migration_translates_for_both_dialects(self) -> None:
        for migration in discover():
            for statement in migration.statements():
                assert to_mysql(statement).strip()
                assert to_sqlite(statement).strip()


class TestSeeding:
    def test_roles_are_seeded(self, repos) -> None:
        assert {r["name"] for r in repos.roles.list_roles()} == {
            "administrator", "ai_engineer", "researcher", "viewer"
        }

    def test_permissions_are_seeded(self, repos) -> None:
        assert repos.database.scalar("SELECT COUNT(*) AS n FROM permissions") == 20

    def test_role_permissions_are_seeded(self, repos) -> None:
        assert repos.database.scalar("SELECT COUNT(*) AS n FROM role_permissions") == 49

    def test_primary_expression_classes_are_active(self, repos) -> None:
        active = {
            str(r["name"])
            for r in repos.database.query(
                "SELECT name FROM emotion_classes WHERE is_active = 1"
            )
        }
        assert active == {
            "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"
        }

    def test_extended_classes_are_present_but_inactive(self, repos) -> None:
        total = repos.database.scalar("SELECT COUNT(*) AS n FROM emotion_classes")
        assert total == 47
        inactive = repos.database.scalar(
            "SELECT COUNT(*) AS n FROM emotion_classes WHERE is_active = 0"
        )
        assert inactive == 40

    def test_seeding_is_idempotent(self, database: Database) -> None:
        migrator = Migrator(database)
        migrator.migrate()
        first = migrator.seed()
        assert sum(first.values()) > 0
        assert sum(migrator.seed().values()) == 0

    def test_settings_are_seeded(self, repos) -> None:
        assert repos.settings.get("app.theme") == "dark"


class TestPasswords:
    def test_hash_is_not_the_password(self) -> None:
        encoded = hash_password(STRONG_PASSWORD)
        assert STRONG_PASSWORD not in encoded
        assert encoded.startswith("pbkdf2_sha256$")

    def test_verify_accepts_the_right_password(self) -> None:
        assert verify_password(STRONG_PASSWORD, hash_password(STRONG_PASSWORD)) is True

    def test_verify_rejects_the_wrong_password(self) -> None:
        assert verify_password("Wrong-Password-2026!", hash_password(STRONG_PASSWORD)) is False

    def test_hashes_are_salted(self) -> None:
        assert hash_password(STRONG_PASSWORD) != hash_password(STRONG_PASSWORD)

    def test_malformed_hash_does_not_raise(self) -> None:
        assert verify_password(STRONG_PASSWORD, "garbage") is False
        assert verify_password(STRONG_PASSWORD, "") is False

    @pytest.mark.parametrize(
        "weak",
        ["short", "alllowercaseletters", "ALLUPPERCASELETTERS", "12345678901234", "nosymbolshere1"],
    )
    def test_weak_passwords_are_rejected(self, weak: str) -> None:
        from visionai.security.passwords import PasswordError

        with pytest.raises(PasswordError):
            hash_password(weak)

    def test_needs_rehash_for_weak_parameters(self) -> None:
        weak = hash_password(STRONG_PASSWORD, iterations=1000)
        assert needs_rehash(weak) is True
        assert needs_rehash(hash_password(STRONG_PASSWORD)) is False

    def test_admin_has_every_permission(self) -> None:
        assert role_has_permission("administrator", Permission.DATABASE_MANAGE)
        assert role_has_permission("administrator", Permission.USER_MANAGE)

    def test_viewer_is_read_only(self) -> None:
        assert role_has_permission("viewer", Permission.DATASET_READ)
        assert not role_has_permission("viewer", Permission.DATASET_WRITE)
        assert not role_has_permission("viewer", Permission.TRAINING_START)

    def test_researcher_cannot_train(self) -> None:
        assert not role_has_permission("researcher", Permission.TRAINING_START)

    def test_engineer_can_train(self) -> None:
        assert role_has_permission("ai_engineer", Permission.TRAINING_START)

    def test_unknown_role_has_no_permissions(self) -> None:
        assert not role_has_permission("wizard", Permission.DATASET_READ)

    def test_unknown_permission_is_rejected(self) -> None:
        assert not role_has_permission("administrator", "not:a:permission")


class TestUserRepository:
    def test_create_and_read(self, repos, admin: int) -> None:
        user = repos.users.get(admin)
        assert user["username"] == "admin"
        assert user["role"] == "administrator"

    def test_password_hash_is_never_returned(self, repos, admin: int) -> None:
        assert "password_hash" not in repos.users.get(admin)
        assert "password_hash" not in repos.users.find_by_username("admin")

    def test_authenticate_succeeds(self, repos, admin: int) -> None:
        assert repos.users.authenticate("admin", STRONG_PASSWORD)["id"] == admin

    def test_authenticate_rejects_a_wrong_password(self, repos, admin: int) -> None:
        assert repos.users.authenticate("admin", "Nope-Wrong-Pass-2026") is None

    def test_authenticate_rejects_an_unknown_user(self, repos) -> None:
        assert repos.users.authenticate("ghost", STRONG_PASSWORD) is None

    def test_suspended_account_cannot_sign_in(self, repos, admin: int) -> None:
        repos.users.set_status(admin, "suspended")
        assert repos.users.authenticate("admin", STRONG_PASSWORD) is None

    def test_last_login_is_recorded(self, repos, admin: int) -> None:
        repos.users.authenticate("admin", STRONG_PASSWORD)
        assert repos.users.get(admin)["last_login_at"] is not None

    def test_duplicate_username_is_rejected(self, repos, admin: int) -> None:
        with pytest.raises(DatabaseError):
            repos.users.create("admin", "other@example.com", STRONG_PASSWORD)

    def test_duplicate_email_is_rejected(self, repos, admin: int) -> None:
        with pytest.raises(DatabaseError):
            repos.users.create("other", "admin@example.com", STRONG_PASSWORD)

    def test_invalid_email_is_rejected(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.users.create("bob", "not-an-email", STRONG_PASSWORD)

    def test_unknown_role_is_rejected(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.users.create("bob", "b@example.com", STRONG_PASSWORD, role="wizard")

    def test_set_password_then_authenticate(self, repos, admin: int) -> None:
        replacement = "Another-Strong-Pass-2026"
        repos.users.set_password(admin, replacement)
        assert repos.users.authenticate("admin", replacement) is not None
        assert repos.users.authenticate("admin", STRONG_PASSWORD) is None

    def test_set_role_changes_permissions(self, repos, admin: int) -> None:
        assert repos.users.has_permission(admin, Permission.USER_MANAGE)
        repos.users.set_role(admin, "viewer")
        assert not repos.users.has_permission(admin, Permission.USER_MANAGE)

    def test_permissions_for_each_role(self, repos) -> None:
        for role, expected in (
            ("administrator", Permission.DATABASE_MANAGE),
            ("ai_engineer", Permission.TRAINING_START),
            ("researcher", Permission.TESTING_RUN),
            ("viewer", Permission.MODEL_READ),
        ):
            user_id = repos.users.create(
                f"user-{role}", f"{role}@example.com", STRONG_PASSWORD, role=role
            )
            assert repos.users.has_permission(user_id, expected), role

    def test_require_permission_raises(self, repos) -> None:
        viewer = repos.users.create("v", "v@example.com", STRONG_PASSWORD, role="viewer")
        with pytest.raises(DatabaseError):
            repos.users.require_permission(viewer, Permission.USER_MANAGE)

    def test_list_and_paginate(self, repos, admin: int) -> None:
        for index in range(5):
            repos.users.create(f"u{index}", f"u{index}@example.com", STRONG_PASSWORD)
        page = repos.users.list_users(page=1, page_size=3)
        assert len(page["items"]) == 3
        assert page["total"] == 6
        assert page["total_pages"] == 2

    def test_filter_by_role(self, repos, admin: int) -> None:
        repos.users.create("v", "v@example.com", STRONG_PASSWORD, role="viewer")
        assert repos.users.list_users(role="viewer")["total"] == 1

    def test_count_by_role(self, repos, admin: int) -> None:
        counts = repos.users.count_by_role()
        assert counts["administrator"] == 1
        assert counts["viewer"] == 0

    def test_delete(self, repos) -> None:
        user_id = repos.users.create("tmp", "tmp@example.com", STRONG_PASSWORD)
        repos.users.delete(user_id)
        assert repos.users.find_by_username("tmp") is None

    def test_profile_round_trip(self, repos, admin: int) -> None:
        repos.users.save_profile(
            admin, display_name="Tommy", occupation="AI Developer", theme="light"
        )
        profile = repos.users.profile(admin)
        assert profile["display_name"] == "Tommy"
        assert profile["theme"] == "light"

    def test_profile_defaults_when_absent(self, repos, admin: int) -> None:
        assert repos.users.profile(admin)["theme"] == "dark"

    def test_profile_preferences_are_json(self, repos, admin: int) -> None:
        repos.users.save_profile(admin, preferences={"camera": 0, "mirror": True})
        assert repos.users.profile(admin)["preferences"] == {"camera": 0, "mirror": True}


class TestSettingsRepository:
    def test_set_and_get(self, repos) -> None:
        repos.settings.set("app.language", "de")
        assert repos.settings.get("app.language") == "de"

    def test_missing_key_returns_the_default(self, repos) -> None:
        assert repos.settings.get("nope", "fallback") == "fallback"

    def test_overwrite(self, repos) -> None:
        repos.settings.set("app.theme", "light")
        repos.settings.set("app.theme", "dark")
        assert repos.settings.get("app.theme") == "dark"

    def test_as_dict(self, repos) -> None:
        assert "app.theme" in repos.settings.as_dict()


class TestDatasetRepository:
    @pytest.fixture
    def dataset_id(self, repos) -> int:
        return repos.datasets.create("Facial Expressions v1", "/tmp/data", "demo")

    def test_create(self, repos, dataset_id: int) -> None:
        assert repos.datasets.get(dataset_id)["name"] == "Facial Expressions v1"

    def test_duplicate_name_version_is_rejected(self, repos, dataset_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.datasets.create("Facial Expressions v1", "/tmp/data", version=1)

    def test_versions_list(self, repos) -> None:
        repos.datasets.create("d", "/tmp/a", version=1)
        repos.datasets.create("d", "/tmp/a", version=2)
        assert [d["version"] for d in repos.datasets.versions("d")] == [2, 1]

    def test_find_by_name(self, repos, dataset_id: int) -> None:
        assert repos.datasets.find_by_name("Facial Expressions v1")["id"] == dataset_id

    def test_import_folder(self, repos, dataset_id: int, sample_images: Path) -> None:
        result = repos.datasets.import_folder(dataset_id, sample_images)
        assert result["inserted"] == 5
        assert result["skipped"] == 0

    def test_import_is_idempotent(self, repos, dataset_id: int, sample_images: Path) -> None:
        repos.datasets.import_folder(dataset_id, sample_images)
        assert repos.datasets.import_folder(dataset_id, sample_images)["skipped"] == 5

    def test_import_missing_folder_is_rejected(self, repos, dataset_id: int, tmp_path: Path) -> None:
        with pytest.raises(DatabaseError):
            repos.datasets.import_folder(dataset_id, tmp_path / "nope")

    def test_statistics_after_import(self, repos, dataset_id: int, sample_images: Path) -> None:
        repos.datasets.import_folder(dataset_id, sample_images)
        stats = repos.datasets.statistics(dataset_id)
        assert stats["images"] == 5
        assert stats["train"] + stats["val"] + stats["test"] == 5

    def test_assign_splits_is_deterministic(self, repos, dataset_id: int, sample_images: Path) -> None:
        repos.datasets.import_folder(dataset_id, sample_images)
        first = repos.datasets.assign_splits(dataset_id, seed=7)
        second = repos.datasets.assign_splits(dataset_id, seed=7)
        assert first == second

    def test_class_distribution(self, repos, dataset_id: int, sample_images: Path) -> None:
        repos.datasets.import_folder(dataset_id, sample_images)
        images = repos.datasets.images.for_dataset(dataset_id)["items"]
        repos.datasets.labels.set_label(images[0]["id"], "happy")
        repos.datasets.labels.set_label(images[1]["id"], "sad")
        assert repos.datasets.class_distribution(dataset_id) == {"happy": 1, "sad": 1}

    def test_search(self, repos, dataset_id: int) -> None:
        assert repos.datasets.search("Facial")["total"] == 1
        assert repos.datasets.search("nothing")["total"] == 0

    def test_set_status(self, repos, dataset_id: int) -> None:
        repos.datasets.set_status(dataset_id, "ready")
        assert repos.datasets.get(dataset_id)["status"] == "ready"

    def test_invalid_status(self, repos, dataset_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.datasets.set_status(dataset_id, "nonsense")

    def test_delete_cascades_to_images(self, repos, dataset_id: int, sample_images: Path) -> None:
        repos.datasets.import_folder(dataset_id, sample_images)
        repos.datasets.delete(dataset_id)
        assert repos.datasets.db.scalar(
            "SELECT COUNT(*) AS n FROM dataset_images WHERE dataset_id = %s", (dataset_id,)
        ) == 0


class TestDatasetLabels:
    @pytest.fixture
    def image_id(self, repos, sample_images: Path) -> int:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        return repos.datasets.images.for_dataset(dataset_id)["items"][0]["id"]

    def test_set_label(self, repos, image_id: int) -> None:
        label_id = repos.datasets.labels.set_label(image_id, "happy", source="ai")
        assert repos.datasets.labels.for_image(image_id)[0]["id"] == label_id

    def test_label_links_the_class(self, repos, image_id: int) -> None:
        repos.datasets.labels.set_label(image_id, "happy")
        row = repos.datasets.labels.for_image(image_id)[0]
        assert row["class_id"] is not None

    def test_unknown_class_still_records(self, repos, image_id: int) -> None:
        repos.datasets.labels.set_label(image_id, "brand-new-class")
        assert repos.datasets.labels.for_image(image_id)[0]["class_id"] is None

    def test_normalised_bounds_are_enforced(self, repos, image_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.datasets.labels.set_label(image_id, "happy", x_center=1.5)

    def test_invalid_source_is_rejected(self, repos, image_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.datasets.labels.set_label(image_id, "happy", source="guesswork")

    def test_replace_for_image(self, repos, image_id: int) -> None:
        repos.datasets.labels.set_label(image_id, "happy")
        repos.datasets.labels.replace_for_image(image_id)
        assert repos.datasets.labels.for_image(image_id) == []

    def test_rename_class(self, repos, image_id: int) -> None:
        repos.datasets.labels.set_label(image_id, "happy")
        dataset_id = repos.datasets.db.scalar("SELECT dataset_id FROM dataset_images WHERE id = %s", (image_id,))
        repos.datasets.labels.rename_class(dataset_id, "happy", "joy")
        assert repos.datasets.labels.for_image(image_id)[0]["class_name"] == "joy"

    def test_progress(self, repos, image_id: int, sample_images: Path) -> None:
        repos.datasets.labels.set_label(image_id, "happy")
        dataset_id = repos.datasets.db.scalar("SELECT dataset_id FROM dataset_images WHERE id = %s", (image_id,))
        progress = repos.datasets.images.progress(dataset_id)
        assert progress["labelled"] == 1
        assert progress["total"] == 5


class TestDatasetValidator:
    def test_healthy_dataset(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        for image in repos.datasets.images.for_dataset(dataset_id)["items"]:
            repos.datasets.labels.set_label(image["id"], "happy")
        report = repos.validator.validate(dataset_id)
        assert report["images_valid"] == 5
        assert report.status == "ok"

    def test_missing_files_are_reported(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        (sample_images / "sample0.jpg").unlink()
        report = repos.validator.validate(dataset_id)
        assert len(report["missing_files"]) == 1
        assert report.status == "error"

    def test_corrupt_image_is_reported(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        (sample_images / "sample0.jpg").write_bytes(b"not an image at all")
        report = repos.validator.validate(dataset_id)
        assert report.status == "error"

    def test_missing_labels_produce_a_warning(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        report = repos.validator.validate(dataset_id)
        assert len(report["missing_labels"]) == 5
        assert report.status == "warning"

    def test_unknown_class_is_an_error(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        image = repos.datasets.images.for_dataset(dataset_id)["items"][0]
        repos.datasets.labels.set_label(image["id"], "not-a-real-class")
        report = repos.validator.validate(dataset_id)
        assert report["invalid_labels"]
        assert report.status == "error"

    def test_class_imbalance_is_reported(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        images = repos.datasets.images.for_dataset(dataset_id)["items"]
        for index, image in enumerate(images):
            repos.datasets.labels.set_label(
                image["id"], "happy" if index else "sad"
            )
        report = repos.validator.validate(dataset_id)
        assert report["class_imbalance"]
        assert report.status == "warning"

    def test_health_status_is_persisted(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        repos.validator.validate(dataset_id)
        assert repos.datasets.get(dataset_id)["health_status"] in ("ok", "warning", "error")

    def test_describe(self, repos, sample_images: Path) -> None:
        dataset_id = repos.datasets.create("d", str(sample_images))
        report = repos.validator.validate(dataset_id)
        assert "Dataset health" in report.describe()


class TestModelRepository:
    @pytest.fixture
    def model_id(self, repos) -> int:
        return repos.models.create("Expression Model", "expression", "demo")

    def test_create_and_list(self, repos, model_id: int) -> None:
        assert repos.models.get(model_id)["name"] == "Expression Model"
        assert repos.models.list_models()["total"] == 1

    def test_duplicate_name_is_rejected(self, repos, model_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.models.create("Expression Model")

    def test_invalid_task_type(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.models.create("m", "telepathy")

    def test_register_version(self, repos, model_id: int) -> None:
        version_id = repos.models.versions.register(model_id, "v1", "/models/m.onnx")
        assert repos.models.versions.get(version_id)["version"] == "v1"

    def test_duplicate_version_is_rejected(self, repos, model_id: int) -> None:
        repos.models.versions.register(model_id, "v1", "/models/m.onnx")
        with pytest.raises(DatabaseError):
            repos.models.versions.register(model_id, "v1", "/models/other.onnx")

    def test_version_needs_a_path(self, repos, model_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.models.versions.register(model_id, "v1", "  ")

    def test_activation_is_explicit(self, repos, model_id: int) -> None:
        first = repos.models.versions.register(model_id, "v1", "/a.onnx")
        assert repos.models.versions.active_version(model_id) is None
        repos.models.versions.activate(first)
        assert repos.models.versions.active_version(model_id)["id"] == first

    def test_activating_retires_the_previous(self, repos, model_id: int) -> None:
        first = repos.models.versions.register(model_id, "v1", "/a.onnx")
        second = repos.models.versions.register(model_id, "v2", "/b.onnx")
        repos.models.versions.activate(first)
        repos.models.versions.activate(second)
        assert repos.models.versions.get(first)["status"] == "retired"
        assert repos.models.versions.active_version(model_id)["id"] == second

    def test_refuses_to_delete_the_active_version(self, repos, model_id: int) -> None:
        version_id = repos.models.versions.register(model_id, "v1", "/a.onnx")
        repos.models.versions.activate(version_id)
        with pytest.raises(DatabaseError):
            repos.models.versions.delete(version_id)

    def test_delete_an_inactive_version(self, repos, model_id: int) -> None:
        version_id = repos.models.versions.register(model_id, "v1", "/a.onnx")
        repos.models.versions.delete(version_id)
        assert repos.models.versions.for_model(model_id) == []

    def test_class_set_round_trip(self, repos, model_id: int) -> None:
        version_id = repos.models.versions.register(
            model_id, "v1", "/a.onnx", class_set=["happy", "sad"]
        )
        from visionai.database.repositories.base import decode_json

        assert decode_json(repos.models.versions.get(version_id)["class_set"]) == ["happy", "sad"]

    def test_summary(self, repos, model_id: int) -> None:
        version_id = repos.models.versions.register(model_id, "v1", "/a.onnx")
        repos.models.versions.activate(version_id)
        summary = repos.models.summary(model_id)
        assert summary["active_version"] == "v1"
        assert summary["version_count"] == 1

    def test_registry_view(self, repos, model_id: int) -> None:
        version_id = repos.models.versions.register(model_id, "v1", "/a.onnx")
        repos.models.versions.activate(version_id)
        assert repos.models.versions.registry()["total"] == 1


class TestTrainingRuns:
    @pytest.fixture
    def ids(self, repos) -> tuple[int, int]:
        model_id = repos.models.create("m", "expression")
        dataset_id = repos.datasets.create("d", "/tmp/x")
        return model_id, dataset_id

    def test_enqueue(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "run 1")
        assert repos.training.get(run_id)["status"] == "queued"

    def test_config_is_stored_as_json(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "run", {"augment": True})
        assert repos.training.get(run_id)["config"] == {"augment": True}

    def test_unknown_dataset_is_rejected(self, repos, ids: tuple[int, int]) -> None:
        model_id, _ = ids
        with pytest.raises(DatabaseError):
            repos.training.enqueue(model_id, 99999, "run")

    def test_claim_next_queued(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        repos.training.enqueue(model_id, dataset_id, "a")
        repos.training.enqueue(model_id, dataset_id, "b")
        first = repos.training.claim_next_queued()
        assert first["run_name"] == "a"
        assert first["status"] == "running"

    def test_claim_returns_none_when_empty(self, repos) -> None:
        assert repos.training.claim_next_queued() is None

    def test_a_claimed_run_is_not_claimed_twice(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        repos.training.enqueue(model_id, dataset_id, "a")
        repos.training.claim_next_queued()
        second = repos.training.claim_next_queued()
        assert second is None or second["run_name"] != "a"

    def test_status_transitions(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a")
        repos.training.set_status(run_id, "running")
        repos.training.set_status(run_id, "completed")
        assert repos.training.get(run_id)["status"] == "completed"
        assert repos.training.get(run_id)["finished_at"] is not None

    def test_invalid_status(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a")
        with pytest.raises(DatabaseError):
            repos.training.set_status(run_id, "dancing")

    def test_fail_records_the_message(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a")
        repos.training.set_error(run_id, "CUDA out of memory")
        assert repos.training.get(run_id)["error_message"] == "CUDA out of memory"

    def test_record_metrics(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a")
        for epoch in range(1, 4):
            repos.training.record_metric(run_id, epoch, train_loss=1.0 / epoch, val_loss=2.0 / epoch)
        metrics = repos.training.metrics(run_id)
        assert len(metrics) == 3
        assert metrics[0]["epoch"] == 1

    def test_unknown_metric_is_rejected(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a")
        with pytest.raises(DatabaseError):
            repos.training.record_metric(run_id, 1, vibes=1.0)

    def test_progress(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a", epochs=10)
        repos.training.record_metric(run_id, 4, train_loss=0.3)
        progress = repos.training.progress(run_id)
        assert progress["epoch"] == 4
        assert progress["percent"] == 40.0
        assert progress["latest"]["train_loss"] == 0.3

    def test_queue_view(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        repos.training.enqueue(model_id, dataset_id, "a")
        assert repos.training.queue_view()[0]["run_name"] == "a"

    def test_cancelling_frees_the_slot(self, repos, ids: tuple[int, int]) -> None:
        model_id, dataset_id = ids
        run_id = repos.training.enqueue(model_id, dataset_id, "a")
        repos.training.set_status(run_id, "cancelled")
        assert repos.training.claim_next_queued() is None


class TestTestRuns:
    @pytest.fixture
    def version_id(self, repos) -> int:
        model_id = repos.models.create("m", "expression")
        return repos.models.versions.register(model_id, "v1", "/a.onnx")

    def test_start_and_finish(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        repos.tests.finish(run_id, accuracy=91.2, precision=90.8, recall=89.9, f1=90.3)
        run = repos.tests.get(run_id)
        assert run["status"] == "completed"
        assert run["accuracy"] == 91.2

    def test_unknown_version_is_rejected(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.tests.start(99999, "eval")

    def test_unknown_source_is_rejected(self, repos, version_id: int) -> None:
        with pytest.raises(DatabaseError):
            repos.tests.start(version_id, "eval", source_kind="telepathy")

    def test_unknown_metric_is_rejected(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        with pytest.raises(DatabaseError):
            repos.tests.finish(run_id, vibes=1.0)

    def test_results_and_accuracy(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        for actual, predicted in (("happy", "happy"), ("sad", "sad"), ("happy", "sad")):
            repos.tests.results.add(run_id, f"/img/{actual}{predicted}.jpg", actual, predicted)
        assert repos.tests.results.accuracy(run_id) == pytest.approx(200 / 3)

    def test_accuracy_is_none_with_no_results(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        assert repos.tests.results.accuracy(run_id) is None

    def test_confusion_matrix(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        for actual, predicted in (
            ("happy", "happy"), ("happy", "happy"), ("happy", "sad"),
            ("sad", "sad"), ("neutral", "happy"),
        ):
            repos.tests.results.add(run_id, "/x.jpg", actual, predicted)
        matrix = repos.tests.confusion.build_from_results(run_id)
        labels = repos.tests.confusion.labels(run_id)
        happy = labels.index("happy")
        sad = labels.index("sad")
        assert matrix[happy][happy] == 2
        assert matrix[happy][sad] == 1
        assert matrix[sad][sad] == 1

    def test_confusion_matrix_is_empty_without_results(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        assert repos.tests.confusion_matrix(run_id) == []

    def test_per_class_accuracy(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        for actual, predicted in (("happy", "happy"), ("happy", "sad")):
            repos.tests.results.add(run_id, "/x.jpg", actual, predicted)
        repos.tests.confusion.build_from_results(run_id)
        accuracy = repos.tests.confusion.per_class_accuracy(run_id)
        assert accuracy["happy"] == 50.0
        assert accuracy["sad"] == 0.0

    def test_confusion_counts_accumulate(self, repos, version_id: int) -> None:
        run_id = repos.tests.start(version_id, "eval")
        repos.tests.confusion.record(run_id, "happy", "happy")
        repos.tests.confusion.record(run_id, "happy", "happy")
        assert repos.tests.confusion.as_matrix(run_id)[0][0] == 2

    def test_compare_versions_returns_metrics_without_a_verdict(
        self, repos, version_id: int
    ) -> None:
        run_id = repos.tests.start(version_id, "eval")
        repos.tests.finish(run_id, accuracy=88.2, precision=87.9, recall=86.8, f1=87.3)
        rows = repos.tests.compare_versions([version_id])
        assert rows[0]["accuracy"] == 88.2
        assert "best" not in rows[0]

    def test_compare_skips_unknown_versions(self, repos, version_id: int) -> None:
        assert repos.tests.compare_versions([99999]) == []


class TestExperiments:
    def test_create_and_read(self, repos) -> None:
        experiment_id = repos.experiments.create("Facial Expression Test #12", notes="try better augmentation")
        row = repos.experiments.get(experiment_id)
        assert row["name"] == "Facial Expression Test #12"
        assert row["status"] == "draft"

    def test_name_is_required(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.experiments.create("  ")

    def test_config_and_hardware_are_json(self, repos) -> None:
        experiment_id = repos.experiments.create(
            "e", config={"fp16": True}, hardware={"gpu": "rtx"}
        )
        row = repos.experiments.get(experiment_id)
        assert row["config"] == {"fp16": True}
        assert row["hardware"] == {"gpu": "rtx"}

    def test_metrics_are_stored(self, repos) -> None:
        experiment_id = repos.experiments.create("e")
        repos.experiments.set_metric(experiment_id, "accuracy", 91.2, "%")
        assert repos.experiments.metrics(experiment_id) == {"accuracy": 91.2}

    def test_metric_overwrite(self, repos) -> None:
        experiment_id = repos.experiments.create("e")
        repos.experiments.set_metric(experiment_id, "accuracy", 90.0)
        repos.experiments.set_metric(experiment_id, "accuracy", 92.0)
        assert repos.experiments.metrics(experiment_id)["accuracy"] == 92.0

    def test_status_transition(self, repos) -> None:
        experiment_id = repos.experiments.create("e")
        repos.experiments.set_status(experiment_id, "complete")
        assert repos.experiments.get(experiment_id)["status"] == "complete"

    def test_invalid_status(self, repos) -> None:
        experiment_id = repos.experiments.create("e")
        with pytest.raises(DatabaseError):
            repos.experiments.set_status(experiment_id, "vibing")

    def test_list_and_filter(self, repos) -> None:
        first = repos.experiments.create("a")
        repos.experiments.create("b")
        repos.experiments.set_status(first, "complete")
        assert repos.experiments.list_experiments(status="complete")["total"] == 1


class TestDetectionSessions:
    def test_start_and_finish(self, repos) -> None:
        session_id = repos.sessions.start("morning", detector_name="yunet", classifier_name="heuristic")
        repos.sessions.finish(session_id, frame_count=300, max_face_count=3, average_fps=28.0)
        row = repos.sessions.get(session_id)
        assert row["frame_count"] == 300
        assert row["max_face_count"] == 3

    def test_invalid_source_is_rejected(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.sessions.start("x", source_kind="hologram")

    def test_trained_flag_is_recorded(self, repos) -> None:
        trained = repos.sessions.start("a", is_trained_model=True)
        heuristic = repos.sessions.start("b", is_trained_model=False)
        assert repos.sessions.get(trained)["is_trained_model"] == 1
        assert repos.sessions.get(heuristic)["is_trained_model"] == 0

    def test_record_a_pipeline_result(self, repos) -> None:
        from visionai.ai.face_tracker import FaceTracker, reset_id_counter
        from visionai.ai.inference import PipelineResult
        from visionai.ai.models.base import Box, ExpressionResult, FaceDetection

        session_id = repos.sessions.start("run")
        reset_id_counter(1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = FaceTracker()
        detections = [FaceDetection(box=Box(100, 100, 240, 260), score=0.9)]
        expressions = {0: ExpressionResult.from_scores({"happy": 0.9, "sad": 0.1}, model="m")}
        tracks = tracker.update(detections, expressions)
        result = PipelineResult(
            frame=frame, frame_number=1, timestamp=0.0,
            detections=detections, tracks=tracks, expressions=expressions,
            total_ms=12.0,
        )
        repos.sessions.faces.record_pipeline_result(session_id, result)

        faces = repos.sessions.faces.for_frame(session_id, 1)
        assert len(faces) == 1
        assert 0.1 < faces[0]["x_center"] < 0.5
        distribution = repos.sessions.expression_distribution(session_id)
        assert distribution == {"happy": 1}

    def test_expression_distribution(self, repos) -> None:
        session_id = repos.sessions.start("run")
        for track_id, label in ((1, "happy"), (2, "sad"), (3, "happy")):
            detection_id = repos.sessions.faces.add(session_id, 1, track_id, 0.5, 0.5, 0.1, 0.1)
            repos.sessions.faces.expressions.add(detection_id, label, 0.9, is_confident=True)
        assert repos.sessions.expression_distribution(session_id) == {"happy": 2, "sad": 1}

    def test_unconfident_results_are_excluded(self, repos) -> None:
        session_id = repos.sessions.start("run")
        detection_id = repos.sessions.faces.add(session_id, 1, 1, 0.5, 0.5, 0.1, 0.1)
        repos.sessions.faces.expressions.add(detection_id, "happy", 0.2, is_confident=False)
        assert repos.sessions.expression_distribution(session_id) == {}

    def test_track_history(self, repos) -> None:
        session_id = repos.sessions.start("run")
        for frame in (1, 2, 3):
            detection_id = repos.sessions.faces.add(session_id, frame, 1, 0.5, 0.5, 0.1, 0.1)
            repos.sessions.faces.expressions.add(detection_id, "happy", 0.9)
        assert len(repos.sessions.faces.history_for_track(session_id, 1)) == 3

    def test_duplicate_frame_and_track_is_rejected(self, repos) -> None:
        session_id = repos.sessions.start("run")
        repos.sessions.faces.add(session_id, 1, 1, 0.5, 0.5, 0.1, 0.1)
        with pytest.raises(DatabaseError):
            repos.sessions.faces.add(session_id, 1, 1, 0.5, 0.5, 0.1, 0.1)

    def test_summary(self, repos) -> None:
        session_id = repos.sessions.start("run", detector_name="yunet")
        repos.sessions.finish(session_id, frame_count=10, max_face_count=2)
        summary = repos.sessions.summary(session_id)
        assert summary["detector"] == "yunet"
        assert summary["frame_count"] == 10


class TestLogging:
    def test_log_and_query(self, repos) -> None:
        repos.logs.log("INFO", "ai", "model loaded")
        repos.logs.log("ERROR", "camera", "device busy")
        page = repos.logs.query_log()
        assert page["total"] == 2

    def test_filter_by_level(self, repos) -> None:
        repos.logs.log("INFO", "ai", "a")
        repos.logs.log("ERROR", "ai", "b")
        assert repos.logs.query_log(level="ERROR")["total"] == 1

    def test_filter_by_category(self, repos) -> None:
        repos.logs.log("INFO", "ai", "a")
        repos.logs.log("INFO", "training", "b")
        assert repos.logs.query_log(category="training")["total"] == 1

    def test_search(self, repos) -> None:
        repos.logs.log("INFO", "ai", "camera opened")
        assert repos.logs.query_log(search="camera")["total"] == 1

    def test_unknown_level_is_normalised(self, repos) -> None:
        repos.logs.log("CHATTY", "ai", "hmm")
        assert repos.logs.counts_by_level() == {"INFO": 1}

    def test_context_is_json(self, repos) -> None:
        repos.logs.log("INFO", "ai", "x", context={"frames": 3})
        assert repos.logs.query_log()["items"][0]["context_json"] == '{"frames": 3}'

    def test_counts_by_level(self, repos) -> None:
        repos.logs.log("ERROR", "ai", "a")
        repos.logs.log("ERROR", "ai", "b")
        assert repos.logs.counts_by_level() == {"ERROR": 2}

    def test_purge(self, repos) -> None:
        repos.logs.log("INFO", "ai", "a")
        assert repos.logs.purge_older_than("2999-01-01 00:00:00") == 1

    def test_hardware_log(self, repos) -> None:
        repos.hardware_logs.log("temperature", "CPU is hot", severity="warning", sensor_value=91.0)
        assert repos.hardware_logs.recent("temperature")[0]["severity"] == "warning"

    def test_system_metrics(self, repos) -> None:
        for _ in range(3):
            repos.system_metrics.record(
                cpu_percent=32.0, ram_percent=48.0, inference_latency_ms=24.0, face_count=2
            )
        assert len(repos.system_metrics.recent(minutes=60)) == 3
        averages = repos.system_metrics.averages(minutes=15)
        assert averages["cpu"] == 32.0
        assert averages["faces"] == 2.0

    def test_unknown_metric_field_is_rejected(self, repos) -> None:
        with pytest.raises(DatabaseError):
            repos.system_metrics.record(vibes=1.0)

    def test_latest(self, repos) -> None:
        repos.system_metrics.record(cpu_percent=10.0)
        repos.system_metrics.record(cpu_percent=20.0)
        assert repos.system_metrics.latest()["cpu_percent"] == 20.0


class TestAnalytics:
    def test_counters(self, repos, admin: int) -> None:
        from visionai.database.analytics import dashboard_counters

        model_id = repos.models.create("m", "expression")
        version_id = repos.models.versions.register(model_id, "v1", "/a.onnx")
        repos.models.versions.activate(version_id)
        counters = dashboard_counters(repos)
        assert counters["models"] == 1
        assert counters["active_versions"] == 1
        assert counters["users"] == 1
        assert counters["datasets"] == 0

    def test_expression_summary(self, repos) -> None:
        from visionai.database.analytics import expression_summary

        session_id = repos.sessions.start("run")
        for track_id, label in ((1, "happy"), (2, "happy"), (3, "sad")):
            detection_id = repos.sessions.faces.add(
                session_id, 1, track_id, 0.5, 0.5, 0.1, 0.1
            )
            repos.sessions.faces.expressions.add(detection_id, label, 0.9, is_confident=True)
        summary = expression_summary(repos)
        assert summary[0]["expression"] == "happy"
        assert summary[0]["count"] == 2

    def test_recent_activity_merges_sources(self, repos) -> None:
        from visionai.database.analytics import recent_activity

        model_id = repos.models.create("m", "expression")
        dataset_id = repos.datasets.create("d", "/tmp/x")
        repos.training.enqueue(model_id, dataset_id, "run")
        repos.sessions.start("sess")
        kinds = {item["kind"] for item in recent_activity(repos)}
        assert kinds == {"training", "session"}


class TestDatabasePrivacy:
    def test_no_biometric_column_exists(self, migrated: Database) -> None:
        """The schema must not offer anywhere to store a face or an embedding."""
        forbidden = {"image", "embedding", "biometric", "face_image", "face_data", "template"}
        for table in migrated.table_names():
            if table == "sqlite_sequence":
                continue
            columns = {
                str(row["name"]).lower()
                for row in migrated.query(f"PRAGMA table_info({table})")
            }
            assert not (columns & forbidden), f"{table} exposes {columns & forbidden}"

    def test_expression_rows_hold_only_measurements(self, repos) -> None:
        session_id = repos.sessions.start("run")
        detection_id = repos.sessions.faces.add(session_id, 1, 1, 0.5, 0.5, 0.2, 0.3)
        repos.sessions.faces.expressions.add(detection_id, "happy", 0.9)
        rows = repos.database.query(
            "SELECT * FROM expression_results WHERE detection_id = %s", (detection_id,)
        )
        assert set(rows[0]) == {
            "id", "detection_id", "expression", "confidence", "is_confident",
            "is_heuristic", "is_trained_model", "alternative", "recorded_at",
        }


@pytest.fixture
def sample_images(tmp_path: Path) -> Path:
    """A directory of small, genuinely decodable images."""
    import cv2

    folder = tmp_path / "images"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        image = np.full((60, 80, 3), (index * 30) % 255, dtype=np.uint8)
        cv2.imwrite(str(folder / f"sample{index}.jpg"), image)
    return folder


class TestRegressionFixes:
    """Guards for bugs found and fixed during the database build."""

    def test_revoke_works_on_sqlite(self, repos) -> None:
        """It used MySQL-only `DELETE rp FROM ... JOIN`, which SQLite rejects."""
        before = repos.roles.get_role("viewer")
        assert Permission.MODEL_READ.value in before["permissions"]
        repos.roles.revoke("viewer", Permission.MODEL_READ)
        after = repos.roles.get_role("viewer")
        assert Permission.MODEL_READ.value not in after["permissions"]
        assert len(after["permissions"]) == len(before["permissions"]) - 1

    def test_revoke_unknown_role_is_harmless(self, repos) -> None:
        repos.roles.revoke("nonexistent", Permission.MODEL_READ)

    def test_dialect_reflects_the_config(self) -> None:
        from visionai.database.driver import Database as Db

        assert Db(DatabaseConfig()).dialect == "mysql"
        assert Db(DatabaseConfig(sqlite_path=":memory:")).dialect == "sqlite"
        assert Db(DatabaseConfig(sqlite_path="/tmp/x.db")).dialect == "sqlite"

    def test_default_config_targets_mysql(self) -> None:
        config = DatabaseConfig.from_env({})
        assert config.is_sqlite is False
        assert config.sqlite_path is None
        assert config.safe()["dialect"] == "mysql"

    def test_sqlite_env_selects_sqlite(self) -> None:
        config = DatabaseConfig.from_env({"VISIONAI_SQLITE_PATH": "/tmp/t.db"})
        assert config.is_sqlite is True
        assert config.safe()["dialect"] == "sqlite"

    def test_safe_describes_the_right_backend(self) -> None:
        assert DatabaseConfig().safe()["host"] == "127.0.0.1"
        assert DatabaseConfig(sqlite_path="/tmp/t.db").safe()["host"] == "local"

    def test_scan_undocumented_images_uses_the_database(self, repos, sample_images: Path, tmp_path: Path) -> None:
        from visionai.database.repositories.datasets import scan_undocumented_images

        dataset_id = repos.datasets.create("d", str(sample_images))
        repos.datasets.import_folder(dataset_id, sample_images)
        assert scan_undocumented_images(sample_images, repos.database) == []
        extra = sample_images / "brand_new.jpg"
        import cv2

        cv2.imwrite(str(extra), np.zeros((10, 10, 3), np.uint8))
        found = scan_undocumented_images(sample_images, repos.database)
        assert [p.name for p in found] == ["brand_new.jpg"]

    def test_scan_without_a_database_returns_everything(self, sample_images: Path) -> None:
        from visionai.database.repositories.datasets import scan_undocumented_images

        assert len(scan_undocumented_images(sample_images)) == 5

    def test_scan_ignores_a_missing_folder(self, tmp_path: Path) -> None:
        from visionai.database.repositories.datasets import scan_undocumented_images

        assert scan_undocumented_images(tmp_path / "nope") == []


class TestHeuristicFeatureIndependence:
    """The brow features must not be the same measurement twice."""

    def test_brow_features_differ(self) -> None:
        from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier

        rng = np.random.default_rng(3)
        classifier = HeuristicExpressionClassifier()
        gaps = []
        for _ in range(20):
            features = classifier.extract_features(
                rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
            )
            gaps.append(abs(features["brow_raise"] - features["brow_furrow"]))
        assert max(gaps) > 0.05, "brow_raise and brow_furrow are the same signal"

    def test_all_seven_classes_stay_reachable(self) -> None:
        from visionai.ai.models.heuristic_classifier import HeuristicExpressionClassifier

        classifier = HeuristicExpressionClassifier()
        features = classifier.extract_features(np.full((96, 96, 3), 170, np.uint8))
        assert set(classifier.score(features)) == {
            "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"
        }

    def test_render_preview_signature_dropped_the_unused_argument(self) -> None:
        import inspect

        from visionai.ai.inference import render_preview

        assert list(inspect.signature(render_preview).parameters) == ["frame", "mirror"]
