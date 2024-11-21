all: pgm.o houghMemConst houghMemGlobal houghMemAll houghMemComp

houghMemConst: houghBaseMemConst.cu pgm.o
	# Compilar el programa con memoria Constante
	nvcc -std=c++14 houghBaseMemConst.cu pgm.o -o houghMemConst -ljpeg \
	-I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu \
	-lopencv_core -lopencv_imgproc -lopencv_imgcodecs

houghMemGlobal: houghBaseMemGlobal.cu pgm.o
	# Compilar el programa con memoria Global
	nvcc -std=c++14 houghBaseMemGlobal.cu pgm.o -o houghMemGlobal -ljpeg \
	-I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu \
	-lopencv_core -lopencv_imgproc -lopencv_imgcodecs

houghMemAll: houghBaseMemAll.cu pgm.o
	# Compilar el programa combinado (Global + Constante + Compartida)
	nvcc -std=c++14 houghBaseMemAll.cu pgm.o -o houghMemAll -ljpeg \
	-I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu \
	-lopencv_core -lopencv_imgproc -lopencv_imgcodecs

houghMemComp: houghBaseMemComp.cu pgm.o
	# Compilar el programa con memoria Compartida
	nvcc -std=c++14 houghBaseMemComp.cu pgm.o -o houghMemComp -ljpeg \
	-I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu \
	-lopencv_core -lopencv_imgproc -lopencv_imgcodecs

pgm.o: common/pgm.cpp
	# Compilar el archivo pgm.cpp
	g++ -std=c++17 -c common/pgm.cpp -o ./pgm.o

clean:
	# Limpiar archivos generados
	rm -f pgm.o houghMemConst houghMemGlobal
