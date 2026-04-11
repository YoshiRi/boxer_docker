SHELL := /bin/bash

.PHONY: bootstrap bootstrap-all build build-gpu compose-config imports smoke shell shell-gpu demo demo-gpu demo1 demo1-gpu demo1-log

bootstrap:
	./scripts/bootstrap_boxer.sh

bootstrap-all:
	./scripts/bootstrap_boxer.sh --all-aria

build:
	docker compose build boxer

build-gpu:
	docker compose --profile gpu build boxer-gpu

compose-config:
	mkdir -p ckpts sample_data output logs
	docker compose config

imports:
	docker compose run --rm boxer python -c "import torch, cv2, dill, tqdm, projectaria_tools; print('imports ok')"

smoke:
	docker compose run --rm boxer python run_boxer.py --help

shell:
	docker compose run --rm boxer bash

shell-gpu:
	docker compose --profile gpu run --rm boxer-gpu bash

demo1:
	docker compose run --rm boxer python run_boxer.py --input nym10_gen1 --max_n=90 --track

demo1-gpu:
	docker compose --profile gpu run --rm boxer-gpu python run_boxer.py --input nym10_gen1 --max_n=90 --track

demo1-log:
	mkdir -p logs
	docker compose run --rm boxer python run_boxer.py --input nym10_gen1 --max_n=90 --track | tee logs/demo1.log

demo:
	$(MAKE) demo1

demo-gpu:
	$(MAKE) demo1-gpu
