all: matrix_multiply run_unit_tests

CC = clang++

PERFORMANCE_PROFILE_DIR = profiles

CPPFLAGS = -Werror -I include -fopencilk -std=c++2a -pthread 
LDFLAGS = -L$(CURDIR)/include -lstdc++ -lm -fopencilk

ifeq ($(DEBUG), 1)
	CPPFLAGS += -Og -g -gdwarf-3 
else
	CPPFLAGS += -Wall -O3 -gdwarf-3 
endif

ifeq ($(CILKSAN),1)
	CFLAGS += -Og -g -fsanitize=cilk -DCILKSAN=1 -D_FORTIFY_SOURCE=0
	LDFLAGS += -fsanitize=cilk
else ifeq ($(CILKSCALE),1)
	CFLAGS  += -fcilktool=cilkscale -DCILKSCALE=1 -O3
	LDFLAGS += -fcilktool=cilkscale
else ifeq ($(PROFILE), 1)
	CFLAGS += -fprofile-generate = $(PERFORMANCE_PROFILE_DIR)
endif


VPATH = shared

MAIN = main.o
OBJS = main.o matrix.o generator.o matrix_printer.o functions.o network_layer.o m_algorithms_concepts.o m_algorithms.o function_object.o function_object_factory.o function_object_iterator.o m_algorithms_utilities.o m_algorithms_register.o matrix_benchmark.o activation_layer.o tensor.o tensor_factory.o tensor_forward_wrapper.o tensor_backwards_pass.o computational_graph_map.o
OBJS_FOR_UNIT_TEST = $(foreach obj, $(OBJS), $(filter-out $(MAIN), $(wildcard *.o))) 


UNIT_TEST_DIRS = ./unittests/
UNIT_TESTING_MAIN = ./unittests/test_all.cpp
UNIT_TESTS_CPP = $(foreach dir,$(UNIT_TEST_DIRS),$(filter-out $(UNIT_TESTING_MAIN), $(wildcard $(dir)*.cpp))) 
UNIT_TESTING_MAIN_OBJ = $(addprefix ./unittests/obj/, $(notdir $(UNIT_TESTING_MAIN:.cpp=.o)))

-include $(OBJS:.o=.d)


%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CPPFLAGS) -c $< -o $@

matrix_multiply: $(OBJS)
	$(CC) -o $@ $(CPPFLAGS) $(OBJS) $(LDFLAGS)

./unittests/obj/test_all.o: $(UNIT_TESTING_MAIN) 
	$(CC) -c $(CPPFLAGS) $(INCLUDE) -o $@ $<

./benchmarks/obj/%.o: ./benchmarks/%.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

%: ./benchmarks/obj/%.o $(OBJS_FOR_UNIT_TEST)
	$(CC) -o $@ $(CPPFLAGS) $^ $(LDFLAGS)

run_unit_tests: $(UNIT_TESTING_MAIN_OBJ) $(UNIT_TESTS_CPP) $(OBJS_FOR_UNIT_TEST)    
	$(CC) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)


clean:
	rm -f matrix_multiply *.o *~ core.* *.d run_unit_tests unittests/obj/*.o benchmarks/obj/*.o benchmark_*