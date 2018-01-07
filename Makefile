all:	open-simplex-noise-test

#akCXXFLAGS=-W -Wall -Wextra -O3
CXXFLAGS=-std=c++11 -W -Wall -Wextra -Wno-unused-parameter -g

open-simplex-noise-test:	open-simplex-noise-test.cpp
	g++ ${CXXFLAGS} -o open-simplex-noise-test open-simplex-noise-test.cpp -lpng

clean:
	rm -f open-simplex-noise-test test2d.png test3d.png test4d.png

