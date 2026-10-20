CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Wno-unused-parameter

build/host_test: csrc/awaaz.c csrc/host_test.c csrc/awaaz.h csrc/awaaz_weights.h
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ csrc/awaaz.c csrc/host_test.c -lm

test: build/host_test
	python -m pytest -q

clean:
	rm -rf build

.PHONY: test clean
