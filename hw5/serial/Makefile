SRCS = ./histogram.cpp
EXT  = .cpp
OBJS = $(SRCS:$(EXT)=.o)
EXEC = histogram
CXX  = g++
CXXFLAGS = -Wall
LIBS =

all:$(EXEC)

ifdef debug
CXXFLAGS += -std=c++11 -D__DEBUG__
endif

$(EXEC):$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o:%$(EXT)
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -rf $(EXEC) $(OBJS)
