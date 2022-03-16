CVROOT=/nfs/opt/bdr/OpenCV
LIBS=-L$(CVROOT)/lib $(shell ls $(CVROOT)/lib/libopencv_* | sed 's/^.*libopencv/-lopencv/g' | sed 's/\.so.*//g' | sort | uniq)
INCLUDE=-I$(CVROOT)/include/opencv4

OBJ=harris harris-laplace

all: $(OBJ)

clean:
	rm -f $(OBJ)

harris: harris.cpp
	g++ -g -Wall $(INCLUDE) -o harris harris.cpp $(LIBS)


harris-laplace: harris-laplace.cpp
	g++ -g -Wall $(INCLUDE) -o harris-laplace harris-laplace.cpp $(LIBS)

