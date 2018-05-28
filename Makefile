all: train classify

train:
	python build_codebook.py
	python train.py

classify:
	python classify.py
