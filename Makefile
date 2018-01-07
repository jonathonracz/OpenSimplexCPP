all:	OpenSimplexTest

#CXXFLAGS=-std=c++11 -W -Wall -Wextra -pedantic -Wno-unused-parameter -O3
CXXFLAGS=-std=c++11 -W -Wall -Wextra -pedantic -Wno-unused-parameter -g

OpenSimplexTest:	OpenSimplexTest.cpp
	g++ ${CXXFLAGS} -o OpenSimplexTest OpenSimplexTest.cpp

clean:
	rm -f OpenSimplexTest test2d.tga test3d.tga test4d.tga
