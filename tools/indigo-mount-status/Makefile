###############################################################################
# 
# Makefile for Allsky360 indigo_mount_status
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

CC = gcc
CFLAGS = -Wall -Wextra -O2 -DINDIGO_LINUX -std=gnu11

# indigo server
INDIGO_ROOT ?= ../../../indigo
INDIGO_CFLAGS = -I$(INDIGO_ROOT)/indigo_libs
INDIGO_LIBS = -L$(INDIGO_ROOT)/build/lib -lindigo

LDFLAGS ?=
LDLIBS  ?=

SRCDIR = src
OBJDIR = obj
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
TARGET := indigo_mount_status

CFLAGS  += -I. $(INDIGO_CFLAGS)
LDFLAGS += $(INDIGO_LIBS) -lpthread -lm

all: $(BINDIR)/$(TARGET)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BINDIR)/$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean
