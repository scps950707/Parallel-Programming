SRCS = ./histogram.cpp
EXT  = .cpp
OBJS = $(SRCS:$(EXT)=.o)
EXEC = histogram
CXX  = g++
CXXFLAGS = -Wall -std=c++11
LIBS = 

all:$(EXEC)

ifdef debug
CXXFLAGS += -D__DEBUG__
endif

$(EXEC):$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o:%$(EXT)
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -rf $(EXEC) $(OBJS)
