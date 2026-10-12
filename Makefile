CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11 -fPIC -Isrc
LDFLAGS ?= -lm

.PHONY: all test clean

all: librotor.so

librotor.so: src/bem.c src/bem.h
	$(CC) $(CFLAGS) -shared -o $@ src/bem.c $(LDFLAGS)

test_bem: test_bem.c src/bem.c src/bem.h
	$(CC) $(CFLAGS) -o $@ test_bem.c src/bem.c $(LDFLAGS)

test: test_bem
	./test_bem

clean:
	rm -f librotor.so test_bem