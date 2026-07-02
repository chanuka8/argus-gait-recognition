from pathlib import Path

ROOT = Path("E:/ARGUS_AI")

DIRS = [
    "core",
    "configs",
    "events",
    "models/architectures",
    "models/weights",
    "models/active",
    "models/candidates",
    "models/rollback",
    "models/gallery",
    "preprocessing",
    "pipeline/steps",
    "intelligence",
    "training",
    "evaluation",
    "automation",
    "enrollment",
    "storage",
    "streaming",
    "monitoring",
    "registry",
    "api/routes",
    "tests/unit",
    "tests/integration",
    "utils",
    "scripts",
    "data/casia_cache",
    "data/casia_processed/silhouettes",
    "data/casia_processed/skeletons",
    "data/casia_processed/gei",
    "data/new_input",
    "data/incremental_cache",
    "data/embeddings",
    "data/faiss_store",
    "runs/exp_001",
    "runs/tensorboard",
    "outputs/videos",
    "outputs/debug",
    "outputs/security_logs",
    "outputs/reports",
    "outputs/eval_reports",
    "docs",
]

PY_FILES = [
    "__init__.py",
    "main.py",
    "cli.py",

    "core/__init__.py",
    "core/system.py",
    "core/system_core.py",
    "core/orchestrator.py",
    "core/boot.py",
    "core/config.py",
    "core/context.py",
    "core/logger.py",
    "core/exceptions.py",
    "core/health_check.py",
    "core/system_monitor.py",

    "events/__init__.py",
    "events/event_bus.py",
    "events/event_types.py",
    "events/dispatcher.py",

    "models/architectures/__init__.py",
    "models/architectures/bygait_light.py",
    "models/architectures/gait_encoder.py",
    "models/architectures/losses.py",

    "preprocessing/__init__.py",
    "preprocessing/casia_extractor.py",
    "preprocessing/silhouette_extractor.py",
    "preprocessing/skeleton_extractor.py",
    "preprocessing/gei_builder.py",
    "preprocessing/augmentation.py",
    "preprocessing/dataset_builder.py",

    "pipeline/__init__.py",
    "pipeline/base_pipeline.py",
    "pipeline/pipeline_factory.py",
    "pipeline/inference_pipeline.py",
    "pipeline/speed_controller.py",
    "pipeline/cache_engine.py",
    "pipeline/steps/__init__.py",
    "pipeline/steps/detection.py",
    "pipeline/steps/tracking.py",
    "pipeline/steps/silhouette_step.py",
    "pipeline/steps/feature_extraction.py",
    "pipeline/steps/matching_step.py",

    "intelligence/__init__.py",
    "intelligence/decision_engine.py",
    "intelligence/confidence_scorer.py",
    "intelligence/policy_engine.py",
    "intelligence/alert_manager.py",

    "training/__init__.py",
    "training/trainer.py",
    "training/dataset.py",
    "training/dataloader.py",
    "training/loss_functions.py",
    "training/optimizer.py",
    "training/callbacks.py",
    "training/checkpointer.py",

    "evaluation/__init__.py",
    "evaluation/evaluator.py",
    "evaluation/metrics.py",
    "evaluation/visualizer.py",

    "automation/__init__.py",
    "automation/lifecycle_controller.py",
    "automation/auto_trainer.py",
    "automation/training_queue.py",
    "automation/model_validator.py",
    "automation/model_promoter.py",
    "automation/rollback_manager.py",

    "enrollment/__init__.py",
    "enrollment/folder_watcher.py",
    "enrollment/enrollment_manager.py",
    "enrollment/enrollment_validator.py",
    "enrollment/gallery_updater.py",
    "enrollment/enrollment_queue.py",

    "storage/__init__.py",
    "storage/data_manager.py",
    "storage/vector_store.py",
    "storage/cache_manager.py",
    "storage/dataset_loader.py",
    "storage/lineage_tracker.py",

    "streaming/__init__.py",
    "streaming/stream_engine.py",
    "streaming/buffer_queue.py",
    "streaming/frame_dropper.py",

    "monitoring/__init__.py",
    "monitoring/gpu_tuner.py",
    "monitoring/crash_guard.py",
    "monitoring/metrics_collector.py",
    "monitoring/performance_profiler.py",

    "registry/__init__.py",
    "registry/model_registry.py",
    "registry/experiment_tracker.py",

    "api/__init__.py",
    "api/server.py",
    "api/schemas.py",
    "api/routes/__init__.py",
    "api/routes/inference.py",
    "api/routes/enrollment.py",
    "api/routes/status.py",

    "utils/__init__.py",
    "utils/video.py",
    "utils/helpers.py",
    "utils/queue_utils.py",
    "utils/math_utils.py",
    "utils/io_utils.py",
    "utils/zip_streamer.py",

    "scripts/system_check.py",
    "scripts/preprocess_casia.py",
    "scripts/build_gallery.py",
    "scripts/benchmark.py",

    "tests/conftest.py",
    "tests/unit/__init__.py",
    "tests/unit/test_preprocessing.py",
    "tests/unit/test_model_arch.py",
    "tests/unit/test_training.py",
    "tests/unit/test_pipeline.py",
    "tests/unit/test_evaluation.py",
    "tests/unit/test_vector_store.py",
    "tests/integration/__init__.py",
    "tests/integration/test_enrollment_flow.py",
    "tests/integration/test_end_to_end.py",
]

TEXT_FILES = {
    "VERSION": "0.1.0\n",

    ".env.example": (
        "ARGUS_MODE=inference\n"
        "ARGUS_DEVICE=cpu\n"
        "ARGUS_CAMERA_INDEX=0\n"
        "ARGUS_CONFIDENCE_THRESHOLD=0.50\n"
        "ARGUS_MATCH_THRESHOLD=0.75\n"
        "ARGUS_ENABLE_API=false\n"
        "ARGUS_ENABLE_AUTO_TRAIN=false\n"
    ),

    ".gitignore": (
        "__pycache__/\n"
        "*.pyc\n"
        "venv/\n"
        ".env\n"
        ".vscode/\n"
        "data/*.zip\n"
        "data/**/*.zip\n"
        "models/weights/*\n"
        "models/active/*\n"
        "models/candidates/*\n"
        "models/rollback/*\n"
        "models/gallery/*\n"
        "outputs/*\n"
        "runs/*\n"
        "!**/.gitkeep\n"
        "!**/README.md\n"
    ),

    "requirements.txt": (
        "numpy\n"
        "opencv-python\n"
        "torch\n"
        "torchvision\n"
        "ultralytics\n"
        "pyyaml\n"
        "python-dotenv\n"
        "psutil\n"
        "watchdog\n"
        "faiss-cpu\n"
        "fastapi\n"
        "uvicorn\n"
        "pydantic\n"
        "matplotlib\n"
        "scikit-learn\n"
        "tqdm\n"
    ),

    "requirements-dev.txt": "pytest\nblack\nruff\nmypy\n",

    "README.md": (
        "# ARGUS AI Surveillance System\n\n"
        "Final year project for person detection and gait recognition.\n\n"
        "Entry points:\n"
        "- main.py: launcher\n"
        "- cli.py: command-line interface\n"
        "- core/system.py: system entry controller\n"
        "- core/orchestrator.py: system brain\n"
    ),

    "Makefile": (
        "install:\n\tpip install -r requirements.txt\n\n"
        "check:\n\tpython scripts/system_check.py\n\n"
        "run:\n\tpython main.py\n\n"
        "test:\n\tpytest tests\n"
    ),

    "configs/base.yaml": "project_name: ARGUS\nversion: 0.1.0\ndevice: cpu\nseed: 42\n",
    "configs/inference.yaml": "camera_index: 0\nconfidence_threshold: 0.50\nmatch_threshold: 0.75\n",
    "configs/train.yaml": "epochs: 20\nbatch_size: 8\nlearning_rate: 0.0001\nembedding_dim: 256\n",
    "configs/gpu_profiles.yaml": "cpu:\n  device: cpu\n  batch_size: 4\n",
    "configs/mode_config.yaml": "default_mode: inference\navailable_modes:\n  - inference\n  - preprocess\n  - train\n  - evaluate\n",
    "configs/auto_train.yaml": "enabled: false\nmin_new_samples: 20\nvalidation_rank1_threshold: 0.70\n",
    "configs/logging.yaml": "level: INFO\nlog_to_file: true\nlog_file: outputs/reports/argus.log\n",

    "data/README.md": (
        "# ARGUS Data Folder\n\n"
        "Large datasets are not stored in Git.\n\n"
        "Place CASIA-B ZIP manually here when needed:\n"
        "data/casia_b_raw.zip\n"
    ),

    "data/new_input/README.md": (
        "# New Person Enrollment Format\n\n"
        "Use this format:\n"
        "data/new_input/person_name/video_01.mp4\n"
        "data/new_input/person_name/images/frame_001.jpg\n"
    ),

    "models/weights/README.md": "Place YOLO and gait model weights here manually.\n",
    "models/active/README.md": "Place active gait_model.pth here.\n",
    "models/gallery/README.md": "Runtime gallery files will be generated here.\n",

    "docs/README.md": (
        "# ARGUS Documentation Map\n\n"
        "- architecture.md: system architecture\n"
        "- pipeline_flow.md: data flow\n"
        "- METHODOLOGY.md: project methodology\n"
        "- TESTING_REPORT.md: testing results\n"
    ),
}

DOCS = [
    "docs/architecture.md",
    "docs/pipeline_flow.md",
    "docs/api_reference.md",
    "docs/deployment.md",
    "docs/security_design.md",
    "docs/gpu_optimization.md",
    "docs/mode_design.md",
    "docs/auto_training_design.md",
    "docs/enrollment_flow.md",
    "docs/LITERATURE_REVIEW.md",
    "docs/METHODOLOGY.md",
    "docs/THREAT_MODELING.md",
    "docs/ETHICAL_CONSIDERATIONS.md",
    "docs/TESTING_REPORT.md",
]


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    for directory in DIRS:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    for file in PY_FILES:
        if file.endswith("__init__.py"):
            write_file(ROOT / file, "")
        else:
            write_file(
                ROOT / file,
                '"""ARGUS module. Implementation will be added step by step."""\n',
            )

    for file, content in TEXT_FILES.items():
        write_file(ROOT / file, content)

    for doc in DOCS:
        title = Path(doc).stem.replace("_", " ").title()
        write_file(ROOT / doc, f"# {title}\n")

    for folder in DIRS:
        write_file(ROOT / folder / ".gitkeep", "")

    write_file(
        ROOT / "scripts" / "start_system.bat",
        "@echo off\ncall venv\\Scripts\\activate\npython main.py\n",
    )

    write_file(
        ROOT / "scripts" / "start_system.sh",
        "#!/bin/bash\nsource venv/bin/activate\npython main.py\n",
    )

    print("ARGUS project scaffold created successfully.")
    print("Location: E:/ARGUS_AI")


if __name__ == "__main__":
    main()