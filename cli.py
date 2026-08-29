import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: list[str]) -> int:
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=PROJECT_ROOT,
        )
        return int(result.returncode)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130


def start_background_process(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
    )


def stop_background_process(process) -> None:
    if process is None:
        return

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def health(args=None) -> int:
    print("\nRunning system health check...")
    return run_command(
        [
            sys.executable,
            "scripts/system_check.py",
        ]
    )


def preprocess(args=None) -> int:
    print("\nRunning CASIA-B preprocessing...")
    return run_command(
        [
            sys.executable,
            "scripts/preprocess_casia.py",
        ]
    )


def train(args=None) -> int:
    print("\nRunning model training...")

    command = [
        sys.executable,
        "scripts/train_model.py",
    ]

    if args is not None:
        if getattr(args, "epochs", None) is not None:
            command.extend(["--epochs", str(args.epochs)])

        if getattr(args, "batch_size", None) is not None:
            command.extend(["--batch-size", str(args.batch_size)])

        if getattr(args, "max_classes", None) is not None:
            command.extend(["--max-classes", str(args.max_classes)])

    return run_command(command)


def build_gallery(args=None) -> int:
    print("\nBuilding training/evaluation gallery...")
    return run_command(
        [
            sys.executable,
            "scripts/build_gallery.py",
        ]
    )


def evaluate(args=None) -> int:
    print("\nRunning evaluation...")
    return run_command(
        [
            sys.executable,
            "scripts/evaluate_model.py",
        ]
    )


def benchmark(args=None) -> int:
    print("\nRunning benchmark...")
    return run_command(
        [
            sys.executable,
            "scripts/benchmark.py",
        ]
    )


def auto_enroll(args=None) -> None:
    input_dir = getattr(args, "input", None) or "data/new_input"

    print("\nRunning one-time auto enrollment...")
    run_command(
        [
            sys.executable,
            "scripts/run_auto_enrollment.py",
            "--input",
            input_dir,
        ]
    )


def auto_enroll_watch(args=None) -> None:
    input_dir = getattr(args, "input", None) or "data/new_input"

    print("\nStarting auto enrollment watcher...")
    print(f"Watching: {input_dir}")
    print("Press CTRL+C to stop.")

    run_command(
        [
            sys.executable,
            "scripts/run_auto_enrollment.py",
            "--input",
            input_dir,
            "--watch",
        ]
    )


def recognize_folder(args=None) -> None:
    print("\nRunning folder-based GEI recognition...")

    folder = getattr(args, "folder", None)

    if not folder:
        folder = input("Enter GEI folder path: ").strip()

    if not folder:
        print("Folder path is required.")
        return

    output = getattr(args, "output", None)
    threshold = getattr(args, "threshold", None) or 0.70

    if output is None:
        typed_output = input("Enter output CSV path or press Enter to skip: ").strip()
        output = typed_output or None

    command = [
        sys.executable,
        "scripts/run_folder_recognition.py",
        "--folder",
        folder,
        "--threshold",
        str(threshold),
    ]

    if output:
        command.extend(
            [
                "--output",
                output,
            ]
        )

    run_command(command)


def recognize_video(args=None) -> None:
    print("\nRunning video-file gait recognition...")

    video = getattr(args, "video", None)

    if not video:
        video = input("Enter video path: ").strip()

    if not video:
        print("Video path is required.")
        return

    output = getattr(args, "output", None)
    threshold = getattr(args, "threshold", None) or 0.75
    show = bool(getattr(args, "show", False))

    if output is None:
        typed_output = input("Enter output CSV path or press Enter to skip: ").strip()
        output = typed_output or None

    if not show and getattr(args, "video", None) is None:
        show_input = input("Show video window? y/n: ").strip().lower()
        show = show_input == "y"

    command = [
        sys.executable,
        "scripts/run_video_recognition.py",
        "--video",
        video,
        "--threshold",
        str(threshold),
    ]

    if output:
        command.extend(
            [
                "--output",
                output,
            ]
        )

    if show:
        command.append("--show")

    run_command(command)


def remove_identity(args=None) -> None:
    person_id = getattr(args, "person_id", None)

    if not person_id:
        person_id = input("Enter identity to remove: ").strip()

    if not person_id:
        print("Person ID is required.")
        return

    print(f"\nRemoving identity from live gallery: {person_id}")

    run_command(
        [
            sys.executable,
            "scripts/remove_gallery_identity.py",
            "--person-id",
            person_id,
        ]
    )


def set_status(args=None) -> None:
    person_id = getattr(args, "person_id", None)
    status = getattr(args, "status", None)

    if not person_id:
        person_id = input("Enter identity: ").strip()

    if not status:
        status = input("Enter status ACTIVE/DISABLED/ARCHIVED: ").strip()

    if not person_id or not status:
        print("Person ID and status are required.")
        return

    run_command(
        [
            sys.executable,
            "scripts/set_gallery_identity_status.py",
            "--person-id",
            person_id,
            "--status",
            status,
        ]
    )


def remove_numeric_identities(args=None) -> None:
    dry_run = bool(getattr(args, "dry_run", False))
    confirm = bool(getattr(args, "confirm", False))

    command = [
        sys.executable,
        "scripts/remove_numeric_gallery_identities.py",
    ]

    if dry_run:
        command.append("--dry-run")
        run_command(command)
        return

    if not confirm:
        print("\nThis will remove all numeric CASIA-B IDs from gallery.")
        print("Run with --confirm to apply.")
        print("Example: python cli.py --mode clean-numeric-gallery --confirm")
        return

    run_command(command)


def live(args=None) -> None:
    input_dir = getattr(args, "input", None) or "data/new_input"

    print("\nStarting ARGUS live gait recognition system...")
    print("Auto enrollment watcher will run in parallel.")
    print(f"New folders inside {input_dir} will be enrolled automatically.")
    print("Already enrolled identities will be skipped.")
    print("Press Q in the camera window to stop live recognition.")

    watcher_process = None

    try:
        watcher_process = start_background_process(
            [
                sys.executable,
                "scripts/run_auto_enrollment.py",
                "--input",
                input_dir,
                "--watch",
            ]
        )

        run_command(
            [
                sys.executable,
                "scripts/run_live_recognition.py",
            ]
        )

    finally:
        print("\nStopping auto enrollment watcher...")
        stop_background_process(
            watcher_process,
        )
        print("Live system stopped.")


def system(args=None) -> None:
    print("\n" + "=" * 60)
    print("ARGUS SYSTEM MODE")
    print("=" * 60)

    print("\n[1/2] Health Check")
    health(args)

    print("\n[2/2] Live Recognition With Auto Enrollment Watcher")
    live(args)


def multi_camera(args=None) -> None:
    print("\n" + "=" * 60)
    print("ARGUS MULTI-CAMERA MODE")
    print("=" * 60)

    from pipeline.multi_camera_recognition import (
        MultiCameraRecognitionPipeline,
    )

    try:
        pipeline = MultiCameraRecognitionPipeline()
        pipeline.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print("\nMulti-camera stopped by user.")


def api(args=None) -> None:
    print("\nStarting API server...")
    print("Open: http://127.0.0.1:8000/docs")
    print("Press CTRL+C to stop.")

    run_command(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.server:app",
            "--reload",
        ]
    )

    print("\nAPI server stopped.")


def tests(args=None) -> int:
    print("\nRunning unit tests...")
    return run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
        ]
    )


def integration_tests(args=None) -> int:
    print("\nRunning integration tests...")
    return run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration",
            "-v",
        ]
    )


def security_test(args=None) -> int:
    print("\nRunning security layer test...")
    return run_command(
        [
            sys.executable,
            "scripts/demo_security_layer.py",
        ]
    )


def confidence_test(args=None) -> int:
    print("\nRunning confidence scorer test...")
    return run_command(
        [
            sys.executable,
            "scripts/demo_confidence_scorer.py",
        ]
    )


def visualizer_test(args=None) -> int:
    print("\nGenerating evaluation charts...")
    return run_command(
        [
            sys.executable,
            "scripts/generate_visualizer_charts.py",
        ]
    )


def streaming_test(args=None) -> int:
    print("\nRunning streaming optimization test...")
    return run_command(
        [
            sys.executable,
            "scripts/demo_streaming_optimization.py",
        ]
    )


def full_check(args=None) -> int:
    print("\n" + "=" * 60)
    print("ARGUS FULL CHECK")
    print("=" * 60)

    code = health(args)
    if code != 0:
        print("\n[CRITICAL FAILURE] Health check failed.")
        return code

    code = tests(args)
    if code != 0:
        print("\n[CRITICAL FAILURE] Unit tests failed.")
        return code

    print("\nRunning full pytest suite...")
    code = run_command([sys.executable, "-m", "pytest", "tests", "-v"])
    if code != 0:
        print("\n[CRITICAL FAILURE] Pytest suite failed.")
        return code

    code = benchmark(args)
    if code != 0:
        print("\n[CRITICAL FAILURE] Benchmark run failed.")
        return code

    code = evaluate(args)
    if code != 0:
        print("\n[CRITICAL FAILURE] Evaluation sweep failed.")
        return code

    print("\nARGUS full check completed successfully.")
    return 0


def prepare(args=None) -> int:
    print("\n" + "=" * 60)
    print("ARGUS PREPARE PIPELINE")
    print("=" * 60)

    code = preprocess(args)
    if code != 0:
        print("\n[ERROR] Preprocessing step failed.")
        return code

    if not getattr(args, "confirm", False):
        print("\n[WARNING] --confirm flag is missing. Training step is bypassed safely.")
        print("To run training, please invoke with --confirm.")
        print("Example: python cli.py --mode prepare --confirm")
        return 0

    code = train(args)
    if code != 0:
        print("\n[ERROR] Model training step failed.")
        return code

    code = build_gallery(args)
    if code != 0:
        print("\n[ERROR] Gallery build step failed.")
        return code

    code = evaluate(args)
    if code != 0:
        print("\n[ERROR] Evaluation step failed.")
        return code

    code = benchmark(args)
    if code != 0:
        print("\n[ERROR] Benchmark step failed.")
        return code

    print("\nARGUS prepare pipeline completed successfully.")
    return 0


def research_eval(args=None) -> int:
    print("\n" + "=" * 60)
    print("ARGUS RESEARCH EVALUATION")
    print("=" * 60)

    code = evaluate(args)
    if code != 0:
        print("\n[ERROR] Evaluation model failed.")
        return code

    print("\nRunning evaluation threshold sweep...")
    code = run_command([sys.executable, "scripts/evaluate_threshold_sweep.py"])
    if code != 0:
        print("\n[ERROR] Threshold sweep failed.")
        return code

    print("\nRunning open-set evaluation...")
    code = run_command([sys.executable, "scripts/evaluate_open_set.py"])
    if code != 0:
        print("\n[ERROR] Open-set evaluation failed.")
        return code

    print("\nRunning cross-view evaluation...")
    code = run_command([sys.executable, "scripts/evaluate_cross_view.py"])
    if code != 0:
        print("\n[ERROR] Cross-view evaluation failed.")
        return code

    print("\nARGUS research evaluation completed successfully. All reports saved.")
    return 0


def production_test(args=None) -> int:
    print("\n" + "=" * 60)
    print("ARGUS PRODUCTION ENVIRONMENT TEST")
    print("=" * 60)

    code = health(args)
    if code != 0:
        print("\n[ERROR] Health check failed.")
        return code

    code = tests(args)
    if code != 0:
        print("\n[ERROR] Unit tests failed.")
        return code

    print("\nRunning full pytest suite...")
    code = run_command([sys.executable, "-m", "pytest", "tests", "-v"])
    if code != 0:
        print("\n[ERROR] Pytest suite failed.")
        return code

    code = benchmark(args)
    if code != 0:
        print("\n[ERROR] Benchmark check failed.")
        return code

    cameras_cfg = Path("configs/cameras.yaml")
    if not cameras_cfg.exists():
        print(f"[ERROR] Required configuration file is missing: {cameras_cfg}")
        return 1
    else:
        print("[OK] Verified: configs/cameras.yaml exists.")

    live_gallery_dir = Path("models/live_gallery")
    if not (live_gallery_dir.exists() and live_gallery_dir.is_dir()):
        print(f"[ERROR] Required live gallery folder is missing: {live_gallery_dir}")
        return 1
    else:
        print("[OK] Verified: models/live_gallery exists.")

    inference_cfg = Path("configs/inference.yaml")
    if not inference_cfg.exists():
        print(f"[ERROR] Required configuration file is missing: {inference_cfg}")
        return 1
    else:
        print("[OK] Verified: configs/inference.yaml exists.")

    print("\nARGUS production environment test completed successfully.")
    return 0


def docs_check(args=None) -> int:
    print("\n" + "=" * 60)
    print("ARGUS DOCUMENTATION VALIDATION")
    print("=" * 60)

    docs_dir = Path("docs")
    if not docs_dir.exists() or not docs_dir.is_dir():
        print(f"[ERROR] Missing documentation directory: {docs_dir}")
        return 1

    required_files = [
        "README.md",
        "requirements.txt",
        "Makefile",
        "docs/README_INDEX.md",
        "docs/matching_person_detection.md",
    ]

    missing = False
    for file_name in required_files:
        path = Path(file_name)
        if path.exists():
            print(f"[OK] Verified: {file_name}")
        else:
            print(f"[ERROR] Missing required file: {file_name}")
            missing = True

    target_folders = [
        "api",
        "configs",
        "core",
        "enrollment",
        "evaluation",
        "events",
        "intelligence",
        "models",
        "monitoring",
        "pipeline",
        "preprocessing",
        "scripts",
        "security_layer",
        "services",
        "storage",
        "streaming",
        "tests",
        "training",
        "utils",
    ]

    for folder in target_folders:
        folder_readme = Path(folder) / "README.md"
        if folder_readme.exists():
            print(f"[OK] Verified package README: {folder_readme}")
        else:
            print(f"[ERROR] Missing package README: {folder_readme}")
            missing = True

    gallery_feat_path = Path("models/gallery/gallery_features.npy")
    gallery_lbl_path = Path("models/gallery/gallery_labels.npy")
    if gallery_feat_path.exists() and gallery_lbl_path.exists():
        try:
            import numpy as np

            _features = np.load(str(gallery_feat_path), allow_pickle=False)
            labels = np.load(str(gallery_lbl_path), allow_pickle=False)
            print(
                f"[OK] Verified gallery metadata: {len(labels)} templates, "
                f"{len(np.unique(labels))} distinct subjects (allow_pickle=False)"
            )
        except (OSError, ValueError) as e:
            print(f"[WARNING] Gallery metadata check notice: {e}")

    index_path = docs_dir / "README_INDEX.md"
    if index_path.exists():
        try:
            index_content = index_path.read_text(encoding="utf-8")
            for folder in target_folders:
                if f"{folder}/README.md" not in index_content:
                    print(f"[ERROR] Missing reference for {folder}/README.md in docs/README_INDEX.md")
                    missing = True
        except OSError as e:
            print(f"[ERROR] Could not read docs/README_INDEX.md: {e}")
            missing = True

    if missing:
        print("\n[ERROR] Documentation verification failed.")
        return 1

    print("\nARGUS documentation check completed successfully (Read-Only).")
    return 0


def demo(args=None) -> None:
    print("\n" + "=" * 60)
    print("ARGUS FINAL END-TO-END DEMO")
    print("=" * 60)

    print("\n[1/8] System Health Check")
    health(args)

    print("\n[2/8] Unit Tests")
    tests(args)

    print("\n[3/8] Integration Tests")
    integration_tests(args)

    print("\n[4/8] Security Layer Validation")
    security_test(args)

    print("\n[5/8] Confidence Scorer Validation")
    confidence_test(args)

    print("\n[6/8] Benchmark")
    benchmark(args)

    print("\n[7/8] One-time Auto Enrollment Check")
    auto_enroll(args)

    print("\n[8/8] Launching Live Recognition With Auto Enrollment")
    live(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARGUS AI Command Line Interface")

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "health",
            "preprocess",
            "train",
            "gallery",
            "evaluate",
            "benchmark",
            "live",
            "system",
            "api",
            "tests",
            "integration-tests",
            "security-test",
            "confidence-test",
            "visualizer-test",
            "streaming-test",
            "full-check",
            "demo",
            "auto-enroll",
            "auto-enroll-watch",
            "recognize-folder",
            "recognize-video",
            "remove-identity",
            "set-status",
            "clean-numeric-gallery",
            "multi-camera",
            "prepare",
            "research-eval",
            "production-test",
            "docs-check",
        ],
        help="ARGUS run mode",
    )

    parser.add_argument(
        "--input",
        default="data/new_input",
        help="Input folder for auto enrollment",
    )

    parser.add_argument(
        "--folder",
        default=None,
        help="Folder path for folder recognition",
    )

    parser.add_argument(
        "--video",
        default=None,
        help="Video path for video recognition",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Recognition threshold",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show video window when supported",
    )

    parser.add_argument(
        "--person-id",
        default=None,
        help="Person identity for gallery maintenance",
    )

    parser.add_argument(
        "--status",
        default=None,
        help="Gallery identity status: ACTIVE, DISABLED, ARCHIVED",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview gallery cleanup without applying changes",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive gallery maintenance action",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )

    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Maximum CASIA-B classes for staged training",
    )

    return parser


def main() -> None:
    from deployment.shutdown_manager import get_shutdown_manager

    get_shutdown_manager().register_signal_handlers()

    parser = build_parser()
    args = parser.parse_args()

    modes = {
        "health": health,
        "preprocess": preprocess,
        "train": train,
        "gallery": build_gallery,
        "evaluate": evaluate,
        "benchmark": benchmark,
        "live": live,
        "system": system,
        "api": api,
        "tests": tests,
        "integration-tests": integration_tests,
        "security-test": security_test,
        "confidence-test": confidence_test,
        "visualizer-test": visualizer_test,
        "streaming-test": streaming_test,
        "full-check": full_check,
        "demo": demo,
        "auto-enroll": auto_enroll,
        "auto-enroll-watch": auto_enroll_watch,
        "recognize-folder": recognize_folder,
        "recognize-video": recognize_video,
        "remove-identity": remove_identity,
        "set-status": set_status,
        "clean-numeric-gallery": remove_numeric_identities,
        "multi-camera": multi_camera,
        "prepare": prepare,
        "research-eval": research_eval,
        "production-test": production_test,
        "docs-check": docs_check,
    }

    code = modes[args.mode](args)
    if isinstance(code, int) and code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
