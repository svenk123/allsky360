CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -I../common
LDFLAGS = -lgpiod
TARGET  = wind_direction
SRC     = wind_direction.c

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/bin/$(TARGET)
