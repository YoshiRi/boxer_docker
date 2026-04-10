SHELL := /bin/bash

.PHONY: bootstrap bootstrap-all build build-gpu imports shell shell-gpu demo demo-gpu

bootstrap:
	./scripts/bootstrap_boxer.sh

bootstrap-all:
	./scripts/bootstrap_boxer.sh --all-aria

build:
	docker compose build boxer

build-gpu:
	docker compose --profile gpu build boxer-gpu

imports:
	docker compose run --rm boxer python -c "import torch, cv2, dill, tqdm, projectaria_tools; print('imports ok')"

shell:
	docker compose run --rm boxer bash

shell-gpu:
	docker compose --profile gpu run --rm boxer-gpu bash

demo:
	docker compose run --rm boxer python run_boxer.py --input nym10_gen1 --max_n=90 --track

demo-gpu:
	docker compose --profile gpu run --rm boxer-gpu python run_boxer.py --input nym10_gen1 --max_n=90 --track
