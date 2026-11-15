.PHONY: help train 

CONFIG ?= configs/config.yaml 

help:
	@echo "Available commands:"
	@echo " make train - Train model"
	@echo " make train CONFIG=path/to/config.yaml"

train: 
	python -m src.main --config $(CONFIG) 
	