all:	open-simplex-noise.o	open-simplex-noise-test

CFLAGS=-W -Wall -Wextra -O3
#CFLAGS=-W -Wall -Wextra -g

open-simplex-noise-test:	open-simplex-noise-test.cpp open-simplex-noise.o
	g++ ${CXXFLAGS} -o open-simplex-noise-test open-simplex-noise.o open-simplex-noise-test.cpp -lpng

open-simplex-noise.o:	open-simplex-noise.h open-simplex-noise.cpp Makefile
	g++ ${CXXFLAGS} -c open-simplex-noise.cpp

clean:
	rm -f open-simplex-noise.o open-simplex-noise-test test2d.png test3d.png test4d.png

