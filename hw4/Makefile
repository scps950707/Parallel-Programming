SRCS = wave.cu
EXT  = .cu
OBJS = $(SRCS:$(EXT)=.o)
EXEC = wave
CXX  = nvcc
CXXFLAGS =
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
