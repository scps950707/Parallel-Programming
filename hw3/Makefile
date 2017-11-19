EX1_SRCS = prime.c
EX2_SRCS = integrate.c
EXT = .c
EX1_OBJS = $(EX1_SRCS:$(EXT)=.o)
EX2_OBJS = $(EX2_SRCS:$(EXT)=.o)
EXEC1 = prime
EXEC2 = integrate
CXX = mpicc
CXXFLAGS = -Wall
LIBS = -lm

all: $(EXEC1) $(EXEC2)

ifdef debug
CXXFLAGS += -D__DEBUG__
endif

$(EXEC1): $(EX1_OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(EX1_OBJS) $(LIBS)

$(EXEC2): $(EX2_OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(EX2_OBJS) $(LIBS)

%.o:%$(EXT)
		$(CXX) $(CXXFLAGS) -c $<

clean:
		rm -rf $(EXEC1) $(EXEC2) $(EX1_OBJS) $(EX2_OBJS)
