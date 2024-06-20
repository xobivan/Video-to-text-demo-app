.PHONY: build build-dev run run-dev test

build:
    docker-compose build app

build-dev:
    docker-compose build app-dev

run:
    docker-compose up app

run-dev:
    docker-compose up app-dev

test:
    docker-compose run --rm app-dev pytest
