CC      = gcc
CFLAGS  = -Wall -Wextra -O2
LDFLAGS = -lgpiod
TARGET  = temp_humidity_sensor
SRC     = temp_humidity_sensor.c

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/bin/$(TARGET)
