###############################################################################
#
# Makefile for the keogram tool
#
# Copyright (c) 2025 Sven Kreiensen
#
# You can use this software under the terms of the MIT software license
# (see LICENSE.md)
#
###############################################################################

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=gnu11 -fopenmp
# Code shared with allsky360-startrails (config, time range, image list, JPEG)
COMMON  = ../common
CFLAGS += -I$(COMMON)

SRCDIR  = src
OBJDIR  = obj
BINDIR  = bin
TARGET  = allsky360-keogram

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES)) \
          $(patsubst $(COMMON)/%.c, $(OBJDIR)/common_%.o, $(wildcard $(COMMON)/*.c))

LIBS    = -ljpeg -ljson-c -lm
INSTALLDIR = /opt/allsky360/bin

.PHONY: all clean install

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(wildcard $(SRCDIR)/*.h) $(wildcard $(COMMON)/*.h) | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/common_%.o: $(COMMON)/%.c $(wildcard $(COMMON)/*.h) | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

install: $(BINDIR)/$(TARGET)
	install -m 755 $(BINDIR)/$(TARGET) $(INSTALLDIR)

clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/$(TARGET)
