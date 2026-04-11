SHELL := /bin/bash

.PHONY: prepare-dirs bootstrap bootstrap-all build build-gpu compose-config imports smoke shell shell-gpu demo demo-gpu demo1 demo1-gpu demo1-log

prepare-dirs:
	mkdir -p ckpts sample_data output logs

bootstrap: prepare-dirs
	./scripts/bootstrap_boxer.sh

bootstrap-all: prepare-dirs
	./scripts/bootstrap_boxer.sh --all-aria

build: prepare-dirs
	docker compose build boxer

build-gpu: prepare-dirs
	docker compose --profile gpu build boxer-gpu

compose-config:
	$(MAKE) prepare-dirs
	docker compose config

imports: prepare-dirs
	docker compose run --rm boxer python -c "import torch, cv2, dill, tqdm, projectaria_tools; print('imports ok')"

smoke: prepare-dirs
	docker compose run --rm boxer python run_boxer.py --help

shell: prepare-dirs
	docker compose run --rm boxer bash

shell-gpu: prepare-dirs
	docker compose --profile gpu run --rm boxer-gpu bash

demo1: prepare-dirs
	docker compose run --rm boxer python run_boxer.py --input nym10_gen1 --max_n=90 --track

demo1-gpu: prepare-dirs
	docker compose --profile gpu run --rm boxer-gpu python run_boxer.py --input nym10_gen1 --max_n=90 --track

demo1-log: prepare-dirs
	docker compose run --rm boxer python run_boxer.py --input nym10_gen1 --max_n=90 --track | tee logs/demo1.log

demo:
	$(MAKE) demo1

demo-gpu:
	$(MAKE) demo1-gpu
