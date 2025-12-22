# Compiler and flags
CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=gnu11 -D_XOPEN_SOURCE=700

# Directory structure
SRCDIR  = src
OBJDIR  = obj
BINDIR  = bin
TARGET  = allsky360-location

# Sources and objects
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))

# Libraries
LIBS    = -ljson-c -lm -lc

.PHONY: all clean install

# Default target
all: $(BINDIR)/$(TARGET)

# Link object files to final binary
$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile each .c to .o
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create obj directory if missing
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Create bin directory if missing
$(BINDIR):
	mkdir -p $(BINDIR)

# Install binary to system
install: $(BINDIR)/$(TARGET)
	install -Dm755 $(BINDIR)/$(TARGET) /usr/local/bin/$(TARGET)

# Remove build artifacts
clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/$(TARGET)
