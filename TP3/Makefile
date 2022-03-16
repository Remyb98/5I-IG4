CVROOT=/nfs/opt/bdr/OpenCV
LIBS=-L$(CVROOT)/lib64 $(shell ls $(CVROOT)/lib64/libopencv_* | sed 's/^.*libopencv/-lopencv/g' | sed 's/\.so.*//g' | sort | uniq)
INCLUDE=-I$(CVROOT)/include/opencv4

ALL=trump calibrate3d

all: $(ALL)

trump: trump.cpp
	g++ -g -Wall $(INCLUDE) -o trump trump.cpp $(LIBS)

calibrate3d: calibrate3d.cpp
	g++ -g -Wall $(INCLUDE) -o calibrate3d calibrate3d.cpp $(LIBS)


clean:
	rm -f $(ALL) *.o *~

