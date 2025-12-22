###############################################################################
# 
# Makefile for meteo tool
#
# Copyright (c) 2025 Sven Kreiensen
#
# You can use this software under the terms of the MIT software license
# (see LICENSE.md)
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#
###############################################################################

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -DINDIGO_LINUX -std=gnu11 -D_XOPEN_SOURCE=700

INCLUDE = -I../../include

# Directory structure

SRCDIR  = src
OBJDIR  = obj
BINDIR  = bin
TARGET  = allsky360-meteo

# Sources and objects
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))

LIBS    = -lcurl -ljson-c -lm -lc -lsqlite3
INSTALLDIR = /opt/allsky360/bin

.PHONY: all clean install

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

install: $(BINDIR)/$(TARGET)
	install -m 755 $(BINDIR)/$(TARGET) $(INSTALLDIR)

clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/$(TARGET)
