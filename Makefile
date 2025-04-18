test:
	pytest tests/

run:
	uvicorn app.main:app --reload

build:
	docker build -t fastapi-codeqa .

run-docker:
	docker run -p 8000:8000 fastapi-codeqa

ingest:
	python scripts/ingest_local_repo.py ./codebase

build-docker-compose:
	docker-compose build

run-docker-compose:
	docker-compose up -d

stop-docker-compose:
	docker-compose down

download-model:
	mkdir -p models
	curl -L https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin -o models/ggml-gpt4all-j-v1.3-groovy.bin
