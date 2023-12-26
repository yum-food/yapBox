CC=g++
CFLAGS=-c -O2 -std=c++20
LDFLAGS=-lcurses

EXE=ui
SRCS=ui.cc
OBJS=$(SRCS:.cc=.o)
HDRS=

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o: %.cc %.h
	$(CC) $(CFLAGS) $< -o $@

%.o: %.cc
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	@rm -f $(OBJS) $(EXE)

