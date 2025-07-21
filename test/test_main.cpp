#include <catch2/catch_test_macros.hpp>
#include "tensor/tensor.hpp"

using namespace cppgrad;

TEST_CASE("tensor const 2", "[tensor]") {
    Tensor t = Tensor::full({2, 2}, 5.0f, true);

    REQUIRE(t.shape() == std::vector<size_t>{2, 2});
}


TEST_CASE("Tensor full constructor", "[tensor]") {

    SECTION("this should work") {
        Tensor t = Tensor::full({2, 2}, 5.0f, true);

        REQUIRE(t.shape() == std::vector<size_t>{2, 2});
    }

    SECTION("this should fail") {
        Tensor t = Tensor::full({2, 2}, 5.0f, true);

        REQUIRE(t.shape() == std::vector<size_t>{2, 3});
    }

}
