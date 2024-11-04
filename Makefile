all: pgm.o hough

hough: houghBase.cu pgm.o
	nvcc -std=c++14 houghBase.cu pgm.o -o hough -ljpeg

pgm.o: common/pgm.cpp
	g++ -std=c++17 -c common/pgm.cpp -o ./pgm.o
