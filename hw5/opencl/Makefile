SRCS = ./histogram.cpp
EXT  = .cpp
OBJS = $(SRCS:$(EXT)=.o)
EXEC = histogram
CXX  = g++
CXXFLAGS = -std=c++11 -Wall
LIBS = -lOpenCL
# LIBS = -L/usr/local/cuda-9.0/lib64/ -lOpenCL

all:$(EXEC)

ifdef debug
CXXFLAGS += -D__DEBUG__
endif

$(EXEC):$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o:%$(EXT)
	$(CXX) $(CXXFLAGS) -c $<
	# $(CXX) $(CXXFLAGS) -c $< -I/usr/local/cuda/include

clean:
	rm -rf $(EXEC) $(OBJS)
