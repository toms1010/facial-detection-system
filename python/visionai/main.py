"""Command-line entry point.

Subcommands:

``run``          launch the desktop application (default)
``headless``     run the pipeline in the terminal
``list-cameras`` enumerate detected capture devices
``devices``      report hardware backends and what each one can see
``self-test``    load every component and report what works
``benchmark``    measure FPS, latency and resource usage
``settings``     show, reset or set individual settings
``models``       list the model files found in models/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from visionai import __version__
from visionai.ai.errors import VisionAIError
from visionai.config.paths import PackagePaths, default_config_path
from visionai.config.settings import Settings, load_settings, save_settings
from visionai.utils.logging_setup import configure_logging

LOG = logging.getLogger("visionai.cli")

EPILOG = """
Expression output is an AI estimate from visible facial features. It is not a
measurement of anyone's actual emotional state. See AI_MODEL.md and PRIVACY.md.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionai",
        description=(
            "Real-time face detection and AI-estimated facial expression "
            "classification for Linux desktops."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"linux-ai-vision {__version__}")
    parser.add_argument("--config", type=Path, help="path to a settings JSON file")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override the configured log level",
    )
    parser.add_argument(
        "--no-log-file", action="store_true", help="do not write a log file"
    )
    # Also accepted after the subcommand ("visionai db --sqlite"). The subparser
    # copy suppresses its default so it cannot clobber this value.
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="use a local throwaway store instead of MySQL (for development)",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_cmd = subparsers.add_parser("run", help="launch the desktop application")
    run_cmd.add_argument(
        "--source",
        default="",
        help="camera index, /dev/videoN, a video file, 'synthetic' or 'none'",
    )
    run_cmd.add_argument(
        "--start", action="store_true", help="begin capturing immediately on launch"
    )
    run_cmd.add_argument("--theme", choices=["dark", "light", "system"])
    run_cmd.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a setting, e.g. --set models.detector=yolo",
    )
    _add_model_arguments(run_cmd)

    headless_cmd = subparsers.add_parser(
        "headless", help="run the pipeline without a display"
    )
    headless_cmd.add_argument("--source", default="synthetic")
    headless_cmd.add_argument(
        "--duration", type=float, default=None, help="stop after N seconds"
    )
    headless_cmd.add_argument("--frames", type=int, default=None, help="stop after N frames")
    headless_cmd.add_argument("--no-colour", action="store_true")
    headless_cmd.add_argument("--no-report", action="store_true")
    headless_cmd.add_argument(
        "--dump-json", type=Path, default=None, help="write the final snapshot as JSON"
    )
    headless_cmd.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE"
    )
    _add_model_arguments(headless_cmd)

    subparsers.add_parser("list-cameras", help="list detected capture devices")
    subparsers.add_parser("devices", help="report hardware backends and sensors")
    subparsers.add_parser("models", help="list available model files")
    subparsers.add_parser("self-test", help="verify every component loads")

    bench = subparsers.add_parser("benchmark", help="measure pipeline performance")
    bench.add_argument("--source", default="synthetic")
    bench.add_argument("--frames", type=int, default=200)
    bench.add_argument("--json", action="store_true", help="emit JSON")
    bench.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE"
    )
    _add_model_arguments(bench)

    settings_cmd = subparsers.add_parser("settings", help="inspect or change settings")
    settings_sub = settings_cmd.add_subparsers(dest="settings_command")
    show_parser = settings_sub.add_parser("show", help="print the current settings")
    show_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="preview overrides without saving them",
    )
    settings_sub.add_parser("path", help="print the settings file path")
    settings_sub.add_parser("reset", help="restore the defaults")
    settings_sub.add_parser(
        "set", help="set one or more values"
    ).add_argument("assignments", nargs="+", metavar="PATH=VALUE")

    db_cmd = subparsers.add_parser("db", help="database schema and connection")
    db_sub = db_cmd.add_subparsers(dest="db_command")
    db_sub.add_parser("status", help="show the connection and migration state")
    db_sub.add_parser("migrate", help="apply pending migrations and seed reference data")
    db_sub.add_parser("seed", help="load reference data only")
    _add_db_arguments(db_cmd)

    user_cmd = subparsers.add_parser("user", help="account administration")
    user_sub = user_cmd.add_subparsers(dest="user_command")
    user_sub.add_parser("list", help="list accounts")
    add_user = user_sub.add_parser("add", help="create an account")
    add_user.add_argument("username")
    add_user.add_argument("email")
    add_user.add_argument("--role", default="viewer")
    add_user.add_argument(
        "--password",
        default=None,
        help="read the password from stdin; omit to generate a strong one",
    )
    _add_db_arguments(user_cmd)

    return parser


def _add_db_arguments(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS, not False: a subparser default would overwrite a global
    # `visionai --sqlite <command>` that was parsed before the subcommand.
    parser.add_argument(
        "--sqlite",
        action="store_true",
        default=argparse.SUPPRESS,
        help="use a local throwaway store instead of MySQL (for development)",
    )


def _open_db(args: argparse.Namespace):
    """Open the database, honouring --sqlite and the MYSQL_* environment."""
    from visionai.database.driver import Database, DatabaseConfig
    from visionai.database.migrator import Migrator, require_up_to_date
    from visionai.database.repositories import open_repositories

    config = DatabaseConfig.from_env()
    if getattr(args, "sqlite", False):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.gettempdir()) / "visionai-dev.db"
        config = DatabaseConfig(sqlite_path=str(path))
    database = Database(config)
    Migrator(database).ensure_table()
    return database, Migrator(database), open_repositories(database), require_up_to_date


def _cmd_db(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.database.driver import DatabaseError

    database, migrator, repositories, _ = _open_db(args)
    command = args.db_command or "status"
    try:
        if command == "status":
            info = database.describe()
            status = migrator.status()
            print("Database")
            print(f"  dialect      : {info['dialect']}")
            print(f"  connected    : {'yes' if info['connected'] else 'no'}")
            print(f"  server       : {info['server_version'] or 'unknown'}")
            print(f"  tables       : {info['tables']}")
            print(f"  target       : {info['target']}")
            if info["error"]:
                print(f"  error        : {info['error']}")
            print("\nMigrations")
            print(f"  {status.describe()}")
            if not status.is_up_to_date:
                print("\n  run: visionai db migrate")
            return 0 if info["connected"] else 1

        if command == "migrate":
            result = migrator.migrate()
            seeded = migrator.seed()
            print(f"applied {len(result)} migration(s)")
            for label in result:
                print(f"  + {label}")
            print(f"seeded: {seeded}")
            return 0

        if command == "seed":
            print(f"seeded: {migrator.seed()}")
            return 0
    except DatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        database.close()
    return 1


def _cmd_user(args: argparse.Namespace, settings: Settings) -> int:
    import getpass

    from visionai.database.driver import DatabaseError
    from visionai.security.passwords import PasswordError, generate_password

    database, migrator, repositories, _ = _open_db(args)
    command = args.user_command or "list"
    try:
        migrator.migrate()
        migrator.seed()

        if command == "list":
            page = repositories.users.list_users(page_size=100)
            if not page["items"]:
                print("no accounts exist yet; create one with `visionai user add`")
                return 0
            print(f"{'username':<20} {'email':<30} {'role':<15} status")
            for row in page["items"]:
                print(
                    f"{row['username']:<20} {row['email']:<30} "
                    f"{row['role']:<15} {row['status']}"
                )
            print(f"\n{page['total']} account(s)")
            return 0

        if command == "add":
            password = args.password or getpass.getpass("Password: ") or generate_password()
            generated = args.password is None and not password
            try:
                user_id = repositories.users.create(
                    args.username, args.email, password, role=args.role
                )
            except PasswordError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(f"created user {args.username} (id {user_id}, role {args.role})")
            if generated:
                print(f"generated password: {password}")
                print("store it now; it is not recoverable and is never logged")
            return 0
    except DatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        database.close()
    return 1


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", choices=["yunet", "haar", "yolo", "onnx"])
    parser.add_argument(
        "--classifier", choices=["heuristic", "onnx", "torchscript"]
    )
    parser.add_argument("--detector-weights")
    parser.add_argument("--classifier-weights")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--detection-confidence", type=float)
    parser.add_argument("--emotion-confidence", type=float)
    parser.add_argument("--inference-fps", type=float)


def _parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            LOG.warning("ignoring malformed override %r; expected PATH=VALUE", pair)
            continue
        path, _, raw = pair.partition("=")
        overrides[path.strip()] = _coerce(raw.strip())
    return overrides


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings(getattr(args, "config", None))
    overrides = _parse_overrides(getattr(args, "set", []))
    for name, key in (
        ("detector", "models.detector"),
        ("classifier", "models.classifier"),
        ("detector_weights", "models.detector_weights"),
        ("classifier_weights", "models.classifier_weights"),
        ("device", "models.device"),
        ("detection_confidence", "models.detection_confidence"),
        ("emotion_confidence", "models.emotion_confidence"),
        ("inference_fps", "pipeline.inference_fps"),
        ("theme", "ui.theme"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[key] = value
    if overrides:
        settings = settings.merge_overrides(overrides)
    return settings.validate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "run"

    settings = _settings_from_args(args)
    configure_logging(
        level=args.log_level or settings.logging.level,
        console=not args.no_log_file and settings.logging.console,
        file=settings.logging.file and not args.no_log_file,
        directory=settings.logging.directory,
        max_bytes=settings.logging.max_bytes,
        backups=settings.logging.backups,
    )

    handlers = {
        "run": _cmd_run,
        "headless": _cmd_headless,
        "list-cameras": _cmd_list_cameras,
        "devices": _cmd_devices,
        "models": _cmd_models,
        "self-test": _cmd_self_test,
        "benchmark": _cmd_benchmark,
        "settings": _cmd_settings,
        "db": _cmd_db,
        "user": _cmd_user,
    }
    handler = handlers.get(command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args, settings)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        return 130
    except VisionAIError as exc:
        LOG.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.ui.app import has_display, qt_available, run

    source = args.source
    if not qt_available():
        print(
            "PySide6 is not installed, so the desktop interface is unavailable.\n"
            "Install it with:  pip install 'linux-ai-vision[qt]'\n"
            "Falling back to headless mode.",
            file=sys.stderr,
        )
        from visionai.ui.headless import run_headless

        return run_headless(settings, source_spec=source or "synthetic")
    if not has_display():
        print(
            "No display detected (DISPLAY and WAYLAND_DISPLAY are unset).\n"
            "Use `visionai headless` instead, or run under X11/Wayland.",
            file=sys.stderr,
        )
        return 3
    return run(settings, source_spec=source, start_immediately=args.start or None)


def _cmd_headless(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.ui.headless import run_headless

    code = run_headless(
        settings,
        source_spec=args.source,
        duration=args.duration,
        frames=args.frames,
        colour=not args.no_colour,
        report=not args.no_report,
    )
    if args.dump_json:
        _dump_benchmark(settings, args.dump_json)
    return code


def _cmd_list_cameras(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.camera.camera_manager import enumerate_cameras

    devices = enumerate_cameras()
    if not devices:
        print("No capture devices found under /dev/video*.")
        print("Check that the camera is connected and that you are in the 'video' group.")
        return 1
    print(f"{len(devices)} capture device(s) detected:\n")
    for device in devices:
        marker = " (default)" if device.is_default else ""
        print(f"  {device.describe()}{marker}")
    return 0


def _cmd_devices(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.hardware.hardware_bridge import HardwareBridge, format_snapshot

    bridge = HardwareBridge(settings.hardware, anonymize=True)
    report = bridge.loader.load()
    print(report.describe())
    print()
    try:
        snapshot = bridge.sample()
    except VisionAIError as exc:
        print(f"error: could not sample hardware: {exc}", file=sys.stderr)
        return 2
    for line in format_snapshot(snapshot):
        print(f"  {line}")
    print(f"\n  backend: {snapshot.backend} v{snapshot.version}")
    print(f"  sample time: {snapshot.sample_ms:.2f} ms")
    for note in snapshot.notes:
        print(f"  note: {note}")
    return 0


def _cmd_models(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.ai.models.registry import ModelRegistry

    paths = PackagePaths.discover()
    registry = ModelRegistry(paths.face_models, paths.emotion_models)
    print("Face models (models/face):")
    for path in paths.face_models.glob("*"):
        if path.is_file():
            print(f"  {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    if not any(p.is_file() for p in paths.face_models.glob("*")):
        print("  none - run: python scripts/fetch_models.py")
    print("\nExpression models (models/emotion):")
    models = registry.available_emotion_models()
    for path in models:
        print(f"  {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    if not models:
        print("  none - the untrained heuristic fallback will be used")
        print("  see AI_MODEL.md for how to add a trained model")
    print("\nAvailable detector backends:", ", ".join(registry.available_detectors()))
    print("Available classifier backends:", ", ".join(registry.available_classifiers()))
    return 0


def _cmd_self_test(args: argparse.Namespace, settings: Settings) -> int:
    import numpy as np

    from visionai.ai.inference import InferencePipeline
    from visionai.ai.models.registry import ModelRegistry
    from visionai.camera.frame_source import SyntheticFrameSource
    from visionai.hardware.hardware_bridge import HardwareBridge
    from visionai.pipeline.engine import PipelineEngine

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("linux-ai-vision self-test\n")

    paths = PackagePaths.discover()
    registry = ModelRegistry(paths.face_models, paths.emotion_models)
    try:
        detector, detector_report = registry.build_detector(settings)
        check("detector loads", detector.is_ready, detector_report.describe())
    except VisionAIError as exc:
        check("detector loads", False, str(exc))
        print("\ncannot continue without a detector")
        return 1
    try:
        classifier, classifier_report = registry.build_classifier(settings)
        check("classifier loads", classifier.is_ready, classifier_report.describe())
    except VisionAIError as exc:
        check("classifier loads", False, str(exc))
        return 1

    pipeline = InferencePipeline(
        settings, registry, detector, classifier, detector_report, classifier_report
    )
    source = SyntheticFrameSource(640, 360)
    source.open()
    frame = source.read()
    check("synthetic frame produced", frame is not None and frame.size > 0)
    started = time.perf_counter()
    result = pipeline.process(frame)
    elapsed = (time.perf_counter() - started) * 1000
    check(
        "inference runs",
        result.error == "",
        f"{elapsed:.1f} ms, {result.face_count} face(s), {pipeline.expression_quality_note}",
    )

    try:
        from visionai.ui.overlay import OverlayMetrics, render_overlay

        rendered = render_overlay(
            frame, result.tracks, metrics=OverlayMetrics(fps=30, inference_ms=elapsed)
        )
        check("overlay renders", rendered.shape == frame.shape)
    except Exception as exc:  # noqa: BLE001
        check("overlay renders", False, f"{type(exc).__name__}: {exc}")

    bridge = HardwareBridge(settings.hardware, anonymize=True)
    try:
        snapshot = bridge.sample()
        check(
            "hardware sampled",
            snapshot.ram_total_bytes > 0,
            f"{snapshot.backend} backend, {snapshot.sample_ms:.2f} ms",
        )
    except VisionAIError as exc:
        check("hardware sampled", False, str(exc))

    engine = PipelineEngine(settings, registry=registry, pipeline=pipeline)
    engine.start("synthetic")
    time.sleep(0.6)
    first = engine.latest()
    check("engine produces results", first is not None, engine.status.describe())
    engine.close()
    check("engine stops cleanly", not engine.is_running)
    del np

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


def _cmd_benchmark(args: argparse.Namespace, settings: Settings) -> int:
    from visionai.benchmark import run_benchmark

    report = run_benchmark(settings, source_spec=args.source, frames=args.frames)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.describe())
    return 0


def _dump_benchmark(settings: Settings, path: Path) -> None:
    from visionai.benchmark import run_benchmark

    report = run_benchmark(settings, source_spec="synthetic", frames=60)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _cmd_settings(args: argparse.Namespace, settings: Settings) -> int:
    command = args.settings_command or "show"
    target = getattr(args, "config", None)

    if command == "path":
        print(default_config_path() if target is None else target)
        return 0
    if command == "show":
        print(json.dumps(settings.to_dict(), indent=2, sort_keys=True))
        print(f"\n# {default_config_path()}", file=sys.stderr)
        return 0
    if command == "reset":
        destination = save_settings(Settings(), target)
        print(f"settings reset to defaults in {destination}")
        return 0
    if command == "set":
        updated = settings.merge_overrides(_parse_overrides(args.assignments))
        destination = save_settings(updated, target)
        print(f"settings written to {destination}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
