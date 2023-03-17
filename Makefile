coverage:
	@echo "Running tests with coverage..."
	coverage run -m pytest -v --disable-warnings
	coverage report -m

deps:
	@echo "Installing dependencies..."
	pip install poetry

integration:
	@echo "Running integration test..."
	pytest blacklight/autoML/tests/end_to_end_feed_forward_test.py -v --disable-warnings
