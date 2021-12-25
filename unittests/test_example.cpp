#include "../deps/catch.hpp"



TEST_CASE("Example Test", "[arithmetic]")
{

    /*
    Instantiate user defined types you wish you test.
    */

    int a = 5;
    int b = 5;

    SECTION("Normal Cases")
    {

        REQUIRE(a + b == 10);
    }

}