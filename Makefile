all: matrix_multiply

CC = g++
# CFLAGS = -g -Werror -Wall -I include -I shared/include -I/usr/local/opt/openssl@1.1/include
CFLAGS = -g -Werror -Wall -I include -std=c++1z

# LDFLAGS = -L/usr/local/opt/openssl@1.1/lib -lssl -lcrypto -pthread
LDFLAGS = -L/include 

VPATH = shared

OBJS = main.o mm.o generator.o matrix_printer.o functions.o network_layer.o m_algorithms.o

-include $(OBJS:.o=.d)


matrix_multiply: $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(OBJS) $(LDFLAGS)

%.d: %.c
	@set -e; gcc -MM $(CFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@;
	@[ -s $@ ] || rm -f $@

%.d: %.cpp
	@set -e; $(CC) -MM $(CFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@;
	@[ -s $@ ] || rm -f $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

clean:
	rm -f matrix_multiply *.o *~ core.* *.d