// #define CATCH_CONFIG_MAIN
// #include "../deps/catch.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../deps/doctest.h"

TEST_CASE("Example") {
     auto name = "Bob";
     REQUIRE(name == "Bob");
}

/*
Entry point for `run_unit_tests` Executable.
Please do not write tests here to avoid recompilation of Catch2 Framework.
Instead create a seperate file for each Class.
*/
