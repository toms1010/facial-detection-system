"""Repositories: one class per database table group.

Everything is explicit SQL. There is no ORM, so the statement that reaches the
database is the statement written here, always parameterised.
"""

from __future__ import annotations

from dataclasses import dataclass

from visionai.database.driver import Database
from visionai.database.repositories.base import (
    NotFoundError,
    Repository,
    ValidationError,
    decode_json,
    encode_json,
    now,
)
from visionai.database.repositories.datasets import (
    DatasetHealthReport,
    DatasetImageRepository,
    DatasetLabelRepository,
    DatasetRepository,
    DatasetValidator,
    checksum_file,
)
from visionai.database.repositories.models import (
    ConfusionMatrixRepository,
    ModelRepository,
    ModelVersionRepository,
    TestResultRepository,
    TestRunRepository,
    TrainingRunRepository,
)
from visionai.database.repositories.ops import (
    ApplicationLogRepository,
    DetectionSessionRepository,
    ExperimentRepository,
    ExpressionResultRepository,
    FaceDetectionRepository,
    HardwareLogRepository,
    SystemMetricRepository,
)
from visionai.database.repositories.users import (
    RoleRepository,
    SettingsRepository,
    UserRepository,
)


@dataclass
class Repositories:
    """Every repository, sharing one connection."""

    users: UserRepository
    roles: RoleRepository
    settings: SettingsRepository
    datasets: DatasetRepository
    models: ModelRepository
    training: TrainingRunRepository
    tests: TestRunRepository
    experiments: ExperimentRepository
    sessions: DetectionSessionRepository
    logs: ApplicationLogRepository
    system_metrics: SystemMetricRepository
    hardware_logs: HardwareLogRepository
    database: Database

    @property
    def validator(self) -> DatasetValidator:
        return DatasetValidator(self.datasets)


def open_repositories(database: Database) -> Repositories:
    """Build the repository set for a connection."""
    return Repositories(
        users=UserRepository(database),
        roles=RoleRepository(database),
        settings=SettingsRepository(database),
        datasets=DatasetRepository(database),
        models=ModelRepository(database),
        training=TrainingRunRepository(database),
        tests=TestRunRepository(database),
        experiments=ExperimentRepository(database),
        sessions=DetectionSessionRepository(database),
        logs=ApplicationLogRepository(database),
        system_metrics=SystemMetricRepository(database),
        hardware_logs=HardwareLogRepository(database),
        database=database,
    )


__all__ = [
    "ApplicationLogRepository",
    "ConfusionMatrixRepository",
    "DatasetHealthReport",
    "DatasetImageRepository",
    "DatasetLabelRepository",
    "DatasetRepository",
    "DatasetValidator",
    "DetectionSessionRepository",
    "ExperimentRepository",
    "ExpressionResultRepository",
    "FaceDetectionRepository",
    "HardwareLogRepository",
    "ModelRepository",
    "ModelVersionRepository",
    "NotFoundError",
    "Repositories",
    "Repository",
    "RoleRepository",
    "SettingsRepository",
    "SystemMetricRepository",
    "TestResultRepository",
    "TestRunRepository",
    "TrainingRunRepository",
    "UserRepository",
    "ValidationError",
    "checksum_file",
    "decode_json",
    "encode_json",
    "now",
    "open_repositories",
]
