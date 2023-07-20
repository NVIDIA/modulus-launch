install:
	pip install --upgrade pip && \
		pip install -e .

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	pre-commit run interrogate -a

lint:
	pre-commit run markdownlint -a

license: 
	pre-commit run license -a

doctest:
	coverage run \
		--rcfile='test/coverage.docstring.rc' \
		-m pytest \
		--doctest-modules modulus/ --ignore-glob=*internal*

pytest: 
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest 

pytest-internal:
	cd test/internal && \
		pytest && \
		cd ../../

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --fail-under=20 && \
		coverage html

container-deploy:
	docker build -t modulus-launch:deploy --target deploy -f Dockerfile .

container-ci:
	docker build -t modulus-launch:ci --target ci -f Dockerfile .

container-docs:
	docker build -t modulus-launch:docs --target docs -f Dockerfile .

