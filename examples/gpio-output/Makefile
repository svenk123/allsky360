CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -I../common
LDFLAGS = -lgpiod
TARGET  = gpio_out
SRC     = gpio_out.c

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/bin/$(TARGET)
