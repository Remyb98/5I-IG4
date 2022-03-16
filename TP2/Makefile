CVROOT=/nfs/opt/bdr/OpenCV
LIBS=-L$(CVROOT)/lib $(shell ls $(CVROOT)/lib/libopencv_* | sed 's/^.*libopencv/-lopencv/g' | sed 's/\.so.*//g' | sort | uniq)
INCLUDE=-I$(CVROOT)/include/opencv4


all: sift confusion

sift: sift.cpp
	g++ -g -Wall $(INCLUDE) -o sift sift.cpp $(LIBS)

match: match.cpp
	g++ -g -Wall $(INCLUDE) -o match match.cpp $(LIBS)

confusion: confusion.cpp
	g++ -g -Wall $(INCLUDE) -o confusion confusion.cpp $(LIBS)


clean:
	rm -f sift match confusion *.o *~

