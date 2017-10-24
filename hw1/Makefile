SRCS = pi.cpp
EXT  = .cpp
OBJS = $(SRCS:$(EXT)=.o)
EXEC = pi
CXX  = g++
CXXFLAGS = -Wall -pthread -std=c++11 -O2 -s
DEP  = dep

all:$(EXEC)

ifdef debug
CXXFLAGS += -D__DEBUG__
endif

$(EXEC):$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)

%.o:%$(EXT)
	$(CXX) $(CXXFLAGS) -c $<

dep:
	$(CXX) $(CXXFLAGS) -M $(SRCS) > $(DEP)

clean:
	rm -rf $(EXEC) $(OBJS) $(DEP)
