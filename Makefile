all: build.log

TAG1=chrisoei/llmexp2

build.log: timestamp.txt nvidia-smi.log
	DOCKER_BUILDKIT=1 docker build --progress=plain -t $(TAG1):`cat timestamp.txt` . 2>&1 | tee build.log
	docker tag $(TAG1):`cat timestamp.txt` $(TAG1):latest

push.log: build.log
	docker push $(TAG1):`cat timestamp.txt` 2>&1 | tee push.log
	docker push $(TAG1):latest 2>&1 | tee -a push.log

push: push.log

shell: huggingface_cache.log
	docker run \
		--gpus all \
		-v huggingface_cache:/home/c/.cache/huggingface \
		-v $(HOME)/output:/home/c/output \
		-it \
		$(TAG1):`cat timestamp.txt` \
		/bin/bash

run: huggingface_cache.log
	docker run \
		--gpus all \
		-v huggingface_cache:/home/c/.cache/huggingface \
		-v $(HOME)/output:/home/c/output \
		-it \
		$(TAG1):`cat timestamp.txt`

clean:
	rm -f build.log push.log timestamp.txt

huggingface_cache.log:
	docker volume create huggingface_cache 2>&1 | tee huggingface_cache.log

timestamp.txt: Dockerfile exp2.py galactica-125m.json run_clm.py
	date -u +'%Y-%m-%d_%H%M%Sz' > timestamp.txt

/usr/bin/lz4:
	sudo apt install -y lz4

/usr/bin/rcs:
	sudo apt install -y rcs

nvidia-smi.log:
	nvidia-smi 2>&1 | tee nvidia-smi.log

permissions.log:
	sudo usermod -aG docker $(USER)
