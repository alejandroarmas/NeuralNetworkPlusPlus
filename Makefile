all: matrix_multiply run_unit_tests

CC = clang++

CPPFLAGS = -Werror -I include -fopencilk -std=c++14 -pthread

ifeq ($(DEBUG), 1)
	CPPFLAGS += -O0 -g -gdwarf-3 
else
	CPPFLAGS += -Wall -O3 -gdwarf-3 
endif

LDFLAGS = -L$(CURDIR)/include -lstdc++ -lm -fopencilk

VPATH = shared

MAIN = main.o
OBJS = main.o matrix.o generator.o matrix_printer.o functions.o network_layer.o m_algorithms.o
OBJS_FOR_UNIT_TEST = $(foreach obj, $(OBJS), $(filter-out $(MAIN), $(wildcard *.o))) 


UNIT_TEST_DIRS = ./unittests/
UNIT_TESTING_MAIN = ./unittests/test_all.cpp
UNIT_TESTS_CPP = $(foreach dir,$(UNIT_TEST_DIRS),$(filter-out $(UNIT_TESTING_MAIN), $(wildcard $(dir)*.cpp))) 
UNIT_TESTING_MAIN_OBJ = $(addprefix ./unittests/obj/, $(notdir $(UNIT_TESTING_MAIN:.cpp=.o)))

-include $(OBJS:.o=.d)



matrix_multiply: $(OBJS)
	$(CC) -o $@ $(CPPFLAGS) $(OBJS) $(LDFLAGS)

run_unit_tests: $(UNIT_TESTING_MAIN_OBJ) $(UNIT_TESTS_CPP) $(DEPS_OBJS_FOR_UNIT_TESTING) $(OBJS_FOR_UNIT_TEST)    
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)
./unittests/obj/test_all.o: $(UNIT_TESTING_MAIN) 
	$(CC) -c $(CPPFLAGS) $(INCLUDE) -o $@ $<


%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

%.o: %.c
	gcc $(CPPFLAGS) -c $< -o $@

clean:
	rm -f matrix_multiply *.o *~ core.* *.d run_unit_tests unittests/obj/*.o