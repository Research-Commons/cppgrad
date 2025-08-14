// #define CATCH_CONFIG_MAIN
// #include <catch2/catch_test_macros.hpp>
// #include <catch2/catch_approx.hpp>
// #include "cppgrad/tensor/tensor.hpp"
//
// using namespace Catch;
//
// // Helper to flatten tensor data into std::vector<float>
// static std::vector<float> to_vector(const cppgrad::Tensor& t) {
//     af::array arr = t.data();
//     std::vector<float> out(t.numel());
//     arr.host(out.data());
//     return out;
// }
//
// // Helper to compute scalar result from single-element tensor
// static float to_scalar(const cppgrad::Tensor& t) {
//     REQUIRE(t.numel() == 1);
//     return to_vector(t)[0];
// }
//
// TEST_CASE("Tensor manual construction and from_array_column_major", "[tensor]") {
//     std::vector<size_t> shape = {2, 3};
//     std::vector<float> values = {1, 2, 3, 4, 5, 6};
//     cppgrad::Tensor t1(shape, values, /*requires_grad=*/false);
//     cppgrad::Tensor t2 = cppgrad::Tensor::from_array_column_major(shape, values, true);
//
//     REQUIRE(t1.shape() == shape);
//     REQUIRE(t1.ndim() == 2);
//     REQUIRE(t1.numel() == values.size());
//     //REQUIRE(to_vector(t1) == values);
//     REQUIRE(!t1.requires_grad());
//
//     REQUIRE(t2.shape() == shape);
//     REQUIRE(t2.requires_grad());
//     REQUIRE(to_vector(t2) == values);
// }
//
// TEST_CASE("Factory initializers: zeros, ones, full, randn", "[tensor]") {
//     auto z = cppgrad::Tensor::zeros({3, 2}, true);
//     REQUIRE(z.shape() == std::vector<size_t>{3, 2});
//     for (auto v : to_vector(z)) REQUIRE(v ==  Approx(0.0f));
//
//     auto o = cppgrad::Tensor::ones({2, 2});
//     REQUIRE(o.numel() == 4);
//     for (auto v : to_vector(o)) REQUIRE(v == Approx(1.0f));
//
//     float fill_val = 7.5f;
//     auto f = cppgrad::Tensor::full({2, 3}, fill_val, true);
//     for (auto v : to_vector(f)) REQUIRE(v == Approx(fill_val));
//
//     auto r = cppgrad::Tensor::randn({4}, false);
//     REQUIRE(r.ndim() == 1);
//     REQUIRE(r.numel() == 4);
//     bool anyNonZero = false;
//     for (auto v : to_vector(r)) {
//         REQUIRE(std::isfinite(v));
//         if (v != 0.0f) anyNonZero = true;
//     }
//     REQUIRE(anyNonZero);
// }
//
// TEST_CASE("Basic arithmetic operations and broadcasting", "[tensor]") {
//     auto a = cppgrad::Tensor::full({2,2}, 2.0f);
//     auto b = cppgrad::Tensor::full({2,2}, 3.0f);
//
//     auto sum = a + b;
//     auto diff = b - a;
//     auto prod = a * b;
//     auto quot = b / a;
//
//     std::vector<float> exp_sum(4, 5.0f);
//     std::vector<float> exp_diff(4, 1.0f);
//     std::vector<float> exp_prod(4, 6.0f);
//     std::vector<float> exp_quot(4, 1.5f);
//
//     REQUIRE(to_vector(sum) == exp_sum);
//     REQUIRE(to_vector(diff) == exp_diff);
//     REQUIRE(to_vector(prod) == exp_prod);
//     REQUIRE(to_vector(quot) == exp_quot);
//
//     // scalar ops
//     auto s1 = a + 1.0f;
//     auto s2 = 5.0f - a;
//     auto s3 = a * 0.5f;
//     auto s4 = 4.0f / a;
//     std::vector<float> exp_s1(4, 3.0f);
//     std::vector<float> exp_s2(4, 3.0f);
//     std::vector<float> exp_s3(4, 1.0f);
//     std::vector<float> exp_s4(4, 2.0f);
//     REQUIRE(to_vector(s1) == exp_s1);
//     REQUIRE(to_vector(s2) == exp_s2);
//     REQUIRE(to_vector(s3) == exp_s3);
//     REQUIRE(to_vector(s4) == exp_s4);
//
//     // unary minus
//     auto neg = -a;
//     std::vector<float> exp_neg(4, -2.0f);
//     REQUIRE(to_vector(neg) == exp_neg);
// }
//
// TEST_CASE("Elementwise functions: exp, log, pow", "[tensor]") {
//     auto x = cppgrad::Tensor::full({2}, 2.0f);
//     auto ex = exp(x);
//     auto lg = log(x);
//     auto pw1 = pow(x, 3.0f);
//     auto pw2 = pow(4.0f, x);
//     std::vector<float> v_ex;
//     for (auto v : to_vector(ex)) v_ex.push_back(std::exp(2.0f));
//     for (size_t i = 0; i < 2; ++i) REQUIRE(to_vector(ex)[i] == Approx(v_ex[i]));
//     for (auto v : to_vector(lg)) REQUIRE(v == Approx(std::log(2.0f)));
//     for (auto v : to_vector(pw1)) REQUIRE(v == Approx(std::pow(2.0f, 3.0f)));
//     for (auto v : to_vector(pw2)) REQUIRE(v == Approx(std::pow(4.0f, 2.0f)));
// }
//
// TEST_CASE("Reduction operations: sum, mean, max", "[tensor]") {
//     std::vector<float> vals = {1,2,3,4,5,6};
//     auto t = cppgrad::Tensor({2,3}, vals);
//     auto s_all = t.sum();
//     auto m_all = t.mean();
//     auto mx_all = t.max();
//     REQUIRE(to_scalar(s_all) == Approx(21.0f));
//     REQUIRE(to_scalar(m_all) == Approx(21.0f/6));
//     REQUIRE(to_scalar(mx_all) == Approx(6.0f));
//
//     auto s_dim0 = t.sum(0);
//     //REQUIRE(s_dim0.shape() == std::vector<size_t>{3});
//     std::vector<float> exp_s0 = {1+4,2+5,3+6};
//     REQUIRE(to_vector(s_dim0) == exp_s0);
//
//     auto m_dim1_k = t.mean(1, true);
//     //REQUIRE(m_dim1_k.shape() == std::vector<size_t>{2,1});
//     std::vector<float> exp_m1 = {(1+2+3)/3.0f, (4+5+6)/3.0f};
//     REQUIRE(to_vector(m_dim1_k) == exp_m1);
// }
//
