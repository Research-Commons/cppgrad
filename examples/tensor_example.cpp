// #include <iostream>
// #include <numeric>  // for std::iota
//
// #include "cppgrad/tensor/tensor.hpp"
// #include "cppgrad/tensor/tensorutils.hpp"
// #include "cppgrad/visualizer/visualizer.hpp"
//
// using namespace cppgrad;
//
// // Global example counter for automated numbering
// static int g_example_counter = 1;
//
// // Helper to print a formatted example header
// void printExampleHeader(const std::string &description) {
//     std::cout << "\n=== Example " << g_example_counter++ << ": " << description << " ===\n";
// }
//
// // Helper to describe a tensor (values, shape, numel, dims)
// void describeTensor(const Tensor &t, const std::string &name) {
//     std::cout << name << ":\n";
//     t.print();
//     t.print_pretty();
//     std::cout << "Shape: ";
//     for (auto s : t.shape()) std::cout << s << " ";
//     std::cout << "| Numel: " << t.numel() << " | Dims: " << t.ndim() << "\n";
// }
//
// // Helper to print gradients for given tensors
// template<typename... Ts>
// void printGrads(const Ts &...ts) {
//     (ts.print_grad(), ...);
// }
//
// int main() {
//     af::info();
//
//     // 1. Manual construction
//     printExampleHeader("Manual construction");
//     std::vector<float> values = {1,2,3,4,5,6};
//     Tensor t1({2,3}, values);
//     describeTensor(t1, "t1");
//     // no gradients here
//
//     // 2. Zeros initialization
//     printExampleHeader("Zeros initialization");
//     describeTensor(Tensor::zeros({2,2}), "t2");
//     // no gradients
//
//     // 3. Ones initialization
//     printExampleHeader("Ones initialization");
//     describeTensor(Tensor::ones({2,3}), "t3");
//
//     // 4. Random normal initialization
//     printExampleHeader("Random normal initialization");
//     describeTensor(Tensor::randn({2,2}), "t4");
//
//     // 5. Full constant initialization
//     printExampleHeader("Full constant initialization");
//     describeTensor(Tensor::full({2,2}, 42.0f), "t5");
//
//     // 6. Elementwise addition and multiplication
//     printExampleHeader("Elementwise addition and multiplication");
//     Tensor add = Tensor::full({2,2}, 1.0f) + Tensor::full({2,2}, 2.0f);
//     Tensor mul = Tensor::full({2,2}, 3.0f) * Tensor::full({2,2}, 4.0f);
//     describeTensor(add, "add");
//     describeTensor(mul, "mul");
//     // no gradients
//
//     // 7. e = a * b + d
//     printExampleHeader("e = a * b + d");
//     {
//         Tensor a = Tensor::full({2,2}, 3.0, true);
//         Tensor b = Tensor::full({2,2}, 4.0, true);
//         Tensor d = Tensor::full({2,2}, 2.0, true);
//         Tensor c = a * b;
//         Tensor e = c + d;
//         e.backward();
//         printGrads(a, b, d, c, e);
//         // Expected:
//         // a.grad = [[4,4],[4,4]]
//         // b.grad = [[3,3],[3,3]]
//         // d.grad = [[1,1],[1,1]]
//         // c.grad = [[1,1],[1,1]]
//         // e.grad = 1
//     }
//
//     // 8. z = a * b * c (scalars)
//     printExampleHeader("z = a * b * c (scalars)");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = Tensor::full({}, 3.0, true);
//         Tensor c = Tensor::full({}, 4.0, true);
//         Tensor z = a * b * c;
//         z.backward();
//         printGrads(a, b, c);
//         // Expected: a.grad=12, b.grad=8, c.grad=6
//     }
//
//     // 9. p = (a + b) * b
//     printExampleHeader("p = (a + b) * b");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = Tensor::full({}, 3.0, true);
//         Tensor s = a + b;
//         Tensor p = s * b;
//         p.backward();
//         printGrads(a, b);
//         // Expected: a.grad = 3, b.grad = (a+b) + b*1 = 2+3 + 3 = 8
//     }
//
//     // 10. Gradients before and after backward
//     printExampleHeader("Gradients before and after backward");
//     {
//         Tensor a = Tensor::full({}, 5.0, true);
//         Tensor b = Tensor::full({}, 7.0, true);
//         Tensor z = a + b;
//         std::cout << "Before backward:\n";
//         printGrads(a, b);
//         z.backward();
//         std::cout << "After backward:\n";
//         printGrads(a, b);
//         // Expected before: a.grad=0, b.grad=0
//         // Expected after: a.grad=1, b.grad=1
//     }
//
//     // 11. Reuse x in multiple operations
//     printExampleHeader("Reuse x in multiple operations");
//     {
//         Tensor x = Tensor::full({}, 2.0, true);
//         Tensor y1 = x * x;
//         Tensor y2 = x + x;
//         Tensor z = y1 + y2;
//         z.backward();
//         printGrads(x);
//         // Expected: x.grad = 2*x + 2 = 2*2 + 2 = 6
//     }
//
//     // 12. Constant tensor (no grad)
//     printExampleHeader("Constant tensor (no grad)");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = Tensor::full({}, 3.0, false);
//         Tensor c = a * b;
//         c.backward();
//         printGrads(a);
//         // Expected: a.grad = 3
//         // b.grad should not exist
//     }
//
//     // 13. Intermediate reuse
//     printExampleHeader("Intermediate reuse");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = a * a;
//         Tensor c = b * a;
//         c.backward();
//         printGrads(a);
//         // Expected: a.grad = 2*a*a' + a*a' = 2*2 + 4 = 8? Actually b=a^2 so db/da=2a=4, dc/db=a so chain: dc/da = dc/db*db/da + dc/da (direct)
//         // But simpler: c = a^3, so dc/da = 3*a^2 = 12
//     }
//
//     // 14. Direct definition chain
//     printExampleHeader("Direct definition chain");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = a * a;
//         Tensor c = b * Tensor::full({}, 5.0, true);
//         c.backward();
//         printGrads(a);
//         // Expected: c = 5*a^2 => dc/da = 10*a = 20
//     }
//
//     // 15. Backward called twice warning
//     printExampleHeader("Backward called twice warning");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = Tensor::full({}, 2.0, true);
//         Tensor c = a * b;
//         c.backward();
//         c.backward();  // debug-only warning
//     }
//
//     // 16. Separate backward on components
//     printExampleHeader("Separate backward on components");
//     {
//         Tensor a = Tensor::full({}, 2.0, true);
//         Tensor b = Tensor::full({}, 3.0, true);
//         Tensor c = a * b;
//         c.backward();
//         printGrads(a); // Expected a.grad = 3
//         b.backward();
//         printGrads(a); // Expected a.grad remains 3
//     }
//
//     // 17. Scalar add and mul
//     printExampleHeader("Scalar add and mul");
//     {
//         Tensor a = Tensor::full({2,1}, 2.0, true);
//         Tensor b = a + 5.f;
//         Tensor c = 5.f + b;
//         c.backward();
//         printGrads(a);
//         // Expected a.grad = 1
//         describeTensor(b, "b");
//         describeTensor(c, "c");
//     }
//
//     // 18. Clone without autograd
//     printExampleHeader("Clone without autograd");
//     {
//         Tensor a = Tensor::full({2,2}, 3.0f, true);
//         Tensor b = TensorUtils::clone(a);
//         describeTensor(a, "a");
//         describeTensor(b, "b");
//         // no gradients
//     }
//
//     // 19. Clone with autograd
//     printExampleHeader("Clone with autograd");
//     {
//         Tensor a = Tensor::full({2,2}, 3.0f, true);
//         Tensor b = TensorUtils::clone_with_grad(a);
//         Tensor c = b * 2.0f;
//         c.backward();
//         printGrads(a, b);
//         // Expected a.grad = 2, b.grad = 2
//     }
//
//     // 20. Matrix multiplication
//     printExampleHeader("Matrix multiplication");
//     {
//         Tensor a({2,3}, {1,2,3,4,5,6});
//         Tensor b({3,2}, {7,8,9,10,11,12});
//         auto c = TensorUtils::matmul(a,b);
//         describeTensor(c, "c");
//         // no gradients
//     }
//
//     // 21. 4D tensor row-major test
//     printExampleHeader("4D tensor row-major test");
//     {
//         std::vector<float> vals(4*3*2);
//         std::iota(vals.begin(), vals.end(), 1.0f);
//         Tensor t3({4,3,2}, vals, false);
//         auto A = t3.impl()->data();
//         std::vector<float> host(vals.size());
//         A.host(host.data());
//         std::cout << "host = { ";
//         for (size_t i = 0; i < host.size(); ++i) {
//             std::cout << host[i] << (i+1<host.size()? ", " : " ");
//         }
//         std::cout << "}\n";
//         describeTensor(t3, "t3");
//         // no gradients
//     }
//
//     // 22. Column-major tensor test
//     printExampleHeader("Column-major tensor test");
//     {
//         Tensor t = Tensor::from_array_column_major({2,3}, {1,2,3,4,5,6});
//         describeTensor(t, "t");
//         // no gradients
//     }
//
//     // 23. ((a + b) * (a - b)) / (a * b)
//     printExampleHeader("((a + b) * (a - b)) / (a * b)");
//     {
//         Tensor a = Tensor::full({2,2}, 3.0f, true);
//         Tensor b = Tensor::full({2,2}, 2.0f, true);
//         auto out = ((a + b) * (a - b)) / (a * b);
//         out.backward();
//         printGrads(a, b);
//         // Expected a.grad ≈ 0.7222, b.grad ≈ -1.0833
//     }
//
//     // 24. log + exp + pow composite op
//     printExampleHeader("log + exp + pow composite op");
//     {
//         Tensor a = Tensor::full({2,2}, 3.0f, true);
//         Tensor b = Tensor::full({2,2}, 2.0f, true);
//         auto frac     = ((a + b) * (a - b)) / (a * b);
//         auto logp     = log(frac);
//         auto expp     = exp(-a);
//         auto powp     = pow(b, a);
//         auto out      = logp + expp + powp;
//         out.backward();
//         printGrads(a, b);
//         // Expected a.grad ≈ 6.362, b.grad ≈ 10.7
//
//         Visualizer::save_dot(out, "graph");
//     }
//
//     // 25. Sum over all elements
//     printExampleHeader("Sum over all elements");
//     {
//         Tensor a = Tensor::full({2,2}, 1.0f, true);
//         auto s = a.sum();
//         describeTensor(s, "s");
//         s.backward();
//         printGrads(a);
//         // Expected a.grad = [[1,1],[1,1]]
//     }
//
//     // 26. Sum along dim=0 (keepdim=false)
//     printExampleHeader("Sum along dim=0 (keepdim=false)");
//     {
//         Tensor a({2,2}, {1,2,3,4}, true);
//         auto s = a.sum(0);
//         describeTensor(s, "s");
//         s.backward();
//         printGrads(a);
//         // Expected a.grad = [[1,1],[1,1]]
//     }
//
//     // 27. Sum along dim=1 (keepdim=true)
//     printExampleHeader("Sum along dim=1 (keepdim=true)");
//     {
//         Tensor a({2,2}, {1,2,3,4}, true);
//         auto s = a.sum(1, true);
//         describeTensor(s, "s");
//         s.backward();
//         printGrads(a);
//         // Expected a.grad = [[1,1],[1,1]]
//     }
//
//     // 28. Sum along dim=1 + scale
//     printExampleHeader("Sum along dim=1 (keepdim=false) + scale");
//     {
//         Tensor a({2,3}, {0,1,2,3,4,5}, true);
//         auto s = a.sum(1);
//         describeTensor(s, "s");
//         auto out = s * Tensor::full({2}, 2.0f);
//         out.backward();
//         printGrads(a);
//         // Expected a.grad = [[2,2,2],[2,2,2]]
//     }
//
//     // 29. Mean over all elements
//     printExampleHeader("Mean over all elements");
//     {
//         Tensor a = Tensor::full({2,2}, 1.0f, true);
//         auto m = a.mean();
//         describeTensor(m, "m");
//         m.backward();
//         printGrads(a);
//         // Expected a.grad = [[0.25,0.25],[0.25,0.25]]
//     }
//
//     // 30. Mean along dim=0 (keepdim=false)
//     printExampleHeader("Mean along dim=0 (keepdim=false)");
//     {
//         Tensor a({2,2}, {1,2,3,4}, true);
//         auto m = a.mean(0);
//         describeTensor(m, "m");
//         m.backward();
//         printGrads(a);
//         // Expected a.grad = [[0.5,0.5],[0.5,0.5]]
//     }
//
//     // 31. Mean along dim=1 (keepdim=true)
//     printExampleHeader("Mean along dim=1 (keepdim=true)");
//     {
//         Tensor a({2,2}, {1,2,3,4}, true);
//         auto m = a.mean(1, true);
//         describeTensor(m, "m");
//         m.backward();
//         printGrads(a);
//         // Expected a.grad = [[0.5,0.5],[0.5,0.5]]
//     }
//
//     // 32. Mean along dim=1 + scale
//     printExampleHeader("Mean along dim=1 (keepdim=false) + scale");
//     {
//         Tensor a({2,3}, {0,1,2,3,4,5}, true);
//         auto m = a.mean(1);
//         describeTensor(m, "m");
//         auto out = m * Tensor::full({2}, 3.0f);
//         out.backward();
//         printGrads(a);
//         // Expected a.grad = [[1,1,1],[1,1,1]]
//     }
//
//     // 33. Max over all elements
//     printExampleHeader("Max over all elements");
//     {
//         Tensor a({2,2}, {1,10,1,1}, true);
//         describeTensor(a, "a");
//         auto m = a.max();
//         describeTensor(m, "m");
//         m.backward();
//         printGrads(a);
//         // Expected a.grad = [[0,1],[0,0]]
//     }
//
//     // 34. Max along dim=0 (keepdim=false)
//     printExampleHeader("Max along dim=0 (keepdim=false)");
//     {
//         Tensor a({2,2}, {1,5,3,4}, true);
//         auto m = a.max(0);
//         describeTensor(m, "m");
//         m.backward();
//         printGrads(a);
//         // Expected a.grad = [[0,1],[1,0]]
//     }
//
//     // 35. Max along dim=1 (keepdim=true)
//     printExampleHeader("Max along dim=1 (keepdim=true)");
//     {
//         Tensor a({2,3}, {1,9,5,2,3,6}, true);
//         auto m = a.max(1, true);
//         describeTensor(m, "m");
//         m.backward();
//         printGrads(a);
//         // Expected a.grad = [[0,1,0],[0,0,1]]
//     }
//
//     // 36. Max dim=1 then multiply & shape check
//     printExampleHeader("Max dim=1 then multiply & shape check");
//     {
//         Tensor a({2,3}, {2,4,6,1,8,7}, true);
//         auto m = a.max(1);  // [6,8]
//         describeTensor(m, "m");
//         auto out = m * Tensor::full({2}, 2.0f);
//         out.backward();
//         printGrads(a);
//         // Expected a.grad = [[0,0,2],[0,2,0]]
//
//         Tensor t({2,3}, {1,2,3,4,5,6});
//         auto s_dim0 = t.sum(0);
//         if (s_dim0.shape() == std::vector<size_t>{2}) {
//             std::cout << "Shape check passed\n";
//         }
//     }
//
//     std::cout << "\nAll examples completed.\n";
//     return 0;
// }

#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/backend/cpu_backend.h"

#include <iostream>

#include "cppgrad/backend/cuda_backend.hpp"

using namespace cppgrad;

int main() {

    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CUDA, CUDA::addKernel);

    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CPU::mulKernel);

    Tensor A({2,3}, {1,2,3,  4,5,6}, DeviceType::CUDA);
    Tensor B({2,3}, {6,5,4,  3,2,1}, DeviceType::CUDA);

    Tensor C = A + B;
    std::cout << "A + B = ";
    for (float v : C.data()) std::cout << v << " ";
    std::cout << std::endl;
    // Expected: 7 7 7 7 7 7


    Tensor A1({2,3}, {1,2,3,  4,5,6}, DeviceType::CPU);
    Tensor B1({2,3}, {6,5,4,  3,2,1}, DeviceType::CPU);

    Tensor C1 = A1 * B1;
    std::cout << "A1 * B1 = ";
    for (float v : C1.data()) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}

