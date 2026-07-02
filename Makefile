.PHONY: install install-dev health test pytest train gallery evaluate benchmark live system multi-camera auto-enroll clean format-check full-check prepare research-eval production-test docs-check

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

health:
	python cli.py --mode health

test:
	python cli.py --mode tests

pytest:
	pytest tests -v

train:
	python cli.py --mode train

gallery:
	python cli.py --mode gallery

evaluate:
	python cli.py --mode evaluate

benchmark:
	python cli.py --mode benchmark

live:
	python cli.py --mode live

system:
	python cli.py --mode system

multi-camera:
	python cli.py --mode multi-camera

auto-enroll:
	python cli.py --mode auto-enroll

full-check:
	python cli.py --mode full-check

prepare:
	python cli.py --mode prepare

research-eval:
	python cli.py --mode research-eval

production-test:
	python cli.py --mode production-test

docs-check:
	python cli.py --mode docs-check

clean:
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True) + glob.glob('.pytest_cache')]; [os.remove(f) for f in glob.glob('**/*.pyc', recursive=True)]"

format-check:
	ruff check .
	black --check .
