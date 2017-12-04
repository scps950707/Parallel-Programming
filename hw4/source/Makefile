SRCS = serial_wave.c
EXT  = .c
OBJS = $(SRCS:$(EXT)=.o)
EXEC = serial_wave
CXX  = gcc
CXXFLAGS = -Wall -std=c99
LIBS = -lm

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
