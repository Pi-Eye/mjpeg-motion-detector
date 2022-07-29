#ifndef CATCH_CONFIG_MAIN
#define CATCH_CONFIG_MAIN
#endif

#include <catch2/catch_all.hpp>

TEST_CASE("True = True") { REQUIRE(true == true); }