// #include <catch2/catch_test_macros.hpp>
// #include "cppgrad/tensor/tensor.hpp"
// #include "cppgrad/tensor/tensorutils.hpp"
// #include <catch2/catch_approx.hpp>
//
// using namespace Catch;
//
// // Helpers
// static std::vector<float> to_vector(const af::array& arr) {
//     std::vector<float> out(arr.elements());
//     arr.host(out.data());
//     return out;
// }
// static float to_scalar(const af::array& arr) {
// //    REQUIRE(arr.elements() == 1);
//     return to_vector(arr)[0];
// }
//
// TEST_CASE("Test1: e = a*b + d backward grads", "[autograd]") {
//     auto a = cppgrad::Tensor::full({2,2}, 3.0f, true);
//     auto b = cppgrad::Tensor::full({2,2}, 4.0f, true);
//     auto d = cppgrad::Tensor::full({2,2}, 2.0f, true);
//     auto c = a * b;
//     auto e = c + d;
//     e.backward();
// //    REQUIRE(to_scalar(e.grad()) == Approx(1.0f));
//     REQUIRE(to_scalar(c.grad()) == Approx(1.0f));
//     REQUIRE(to_scalar(a.grad()) == Approx(4.0f));
//     REQUIRE(to_scalar(b.grad()) == Approx(3.0f));
//     REQUIRE(to_scalar(d.grad()) == Approx(1.0f));
// }
//
// TEST_CASE("Test2: z = a*b*c scalar autograd", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto b = cppgrad::Tensor::full({}, 3.0f, true);
//     auto c = cppgrad::Tensor::full({}, 4.0f, true);
//     auto z = a * b * c;
//     z.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(12.0f));
//     REQUIRE(to_scalar(b.grad()) == Approx(8.0f));
//     REQUIRE(to_scalar(c.grad()) == Approx(6.0f));
// }
//
// TEST_CASE("Test3: p = (a+b)*b scalar autograd", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto b = cppgrad::Tensor::full({}, 3.0f, true);
//     auto s = a + b;
//     auto p = s * b;
//     p.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(3.0f));
//     REQUIRE(to_scalar(b.grad()) == Approx(8.0f));
// }
//
// TEST_CASE("Test4: grads before and after backward", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 5.0f, true);
//     auto b = cppgrad::Tensor::full({}, 7.0f, true);
//     auto z = a + b;
//     REQUIRE_NOTHROW(to_scalar(a.grad()));
//     REQUIRE(to_scalar(a.grad()) == Approx(0.0f));
//     REQUIRE(to_scalar(b.grad()) == Approx(0.0f));
//     z.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(1.0f));
//     REQUIRE(to_scalar(b.grad()) == Approx(1.0f));
// }
//
// TEST_CASE("Test5: reuse x in multiple ops", "[autograd]") {
//     auto x = cppgrad::Tensor::full({}, 2.0f, true);
//     auto y1 = x * x;
//     auto y2 = x + x;
//     auto z = y1 + y2;
//     z.backward();
//     REQUIRE(to_scalar(x.grad()) == Approx(6.0f));
// }
//
// TEST_CASE("Test6: constant tensor does not accumulate grad", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto b = cppgrad::Tensor::full({}, 3.0f, false);
//     auto c = a * b;
//     c.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(3.0f));
// //    REQUIRE(to_scalar(b.grad()) == Approx(0.0f));
// }
//
// TEST_CASE("Test7: intermediate reuse scalar", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto b = a * a;
//     auto c = b * a;
//     c.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(12.0f));
// }
//
// TEST_CASE("Test8: direct definition with inline full", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto c = (a * a) * cppgrad::Tensor::full({}, 5.0f, true);
//     c.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(20.0f));
// }
//
// TEST_CASE("Test9: backward called twice warning (debug)", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto b = cppgrad::Tensor::full({}, 2.0f, true);
//     auto c = a * b;
//     c.backward();
//     REQUIRE_NOTHROW(c.backward());
// }
//
// TEST_CASE("Test10: backward independent tensor", "[autograd]") {
//     auto a = cppgrad::Tensor::full({}, 2.0f, true);
//     auto b = cppgrad::Tensor::full({}, 3.0f, true);
//     auto c = a * b;
//     c.backward();
//     REQUIRE(to_scalar(a.grad()) == Approx(3.0f));
//     b.backward();
//     //REQUIRE(to_scalar(a.grad()) == Approx(1.0f + 3.0f));
// }
//
// TEST_CASE("Test12 & 13: scalar add and mul autograd", "[autograd]") {
//     auto a = cppgrad::Tensor::full({2,1}, 2.0f, true);
//     auto b_add = a + 5.0f;
//     auto c_add = 5.0f + b_add;
//     c_add.backward();
//     REQUIRE(to_vector(a.grad()) == std::vector<float>{1.0f, 1.0f});
//     a.zero_grad();
//
//     auto b_mul = a * 5.0f;
//     auto c_mul = 5.0f * b_mul;
//     c_mul.backward();
//     REQUIRE(to_vector(a.grad()) == std::vector<float>{25.0f, 25.0f});
// }
//
// TEST_CASE("Test14 & 15: clone without and with autograd", "[tensor][autograd]") {
//     auto a = cppgrad::Tensor::full({2,2}, 3.0f, true);
//     auto b = cppgrad::TensorUtils::clone(a);
//     REQUIRE_THROWS(b.backward());
//
//     auto b2 = cppgrad::TensorUtils::clone_with_grad(a);
//     auto c = b2 * 2.0f;
//     c.backward();
//     REQUIRE(to_vector(a.grad()) == to_vector(b2.grad()));
// }
//
// // TEST_CASE("Test19 & 20: complex expression gradients", "[autograd]") {
// //     auto a = cppgrad::Tensor::full({2,2}, 3.0f, true);
// //     auto b = cppgrad::Tensor::full({2,2}, 2.0f, true);
// //     auto c = a + b;
// //     auto d = a - b;
// //     auto e = c * d;
// //     auto f = a * b;
// //     auto frac = e / f;
// //     auto logp = log(frac);
// //     auto expp = exp(-a);
// //     auto powp = pow(b,a);
// //     auto out = logp + expp + powp;
// //     out.backward();
// //
// //     for (auto v : to_vector(a.grad())) REQUIRE(std::isfinite(v));
// //     for (auto v : to_vector(b.grad())) REQUIRE(std::isfinite(v));
// // }
