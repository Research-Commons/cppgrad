#include "../include/tensor/tensor.hpp"
#include <iostream>

using namespace cppgrad;

int main() {
    af::info();

    // // Test: Manual construction
    // std::vector<float> values = {1, 2, 3, 4, 5, 6};
    // Tensor t1({2, 3}, values);
    // std::cout << "Tensor t1 (manual):" << std::endl;
    // t1.print();
    //
    // std::cout << "Shape: ";
    // for (auto s : t1.shape()) std::cout << s << " ";
    //
    // std::cout << "\nNumel: " << t1.numel() << ", Dims: " << t1.ndim() << std::endl;
    // std::cout << std::endl;
    //
    // // Test: zeros
    // Tensor t2 = Tensor::zeros({2, 2});
    // std::cout << "Tensor t2 (zeros):" << std::endl;
    // t2.print();
    //
    // // Test: ones
    // Tensor t3 = Tensor::ones({2, 3});
    // std::cout << "Tensor t3 (ones):" << std::endl;
    // t3.print();
    //
    // // Test: randn
    // Tensor t4 = Tensor::randn({2, 2});
    // std::cout << "Tensor t4 (randn):" << std::endl;
    // t4.print();
    //
    // // Test: full
    // Tensor t5 = Tensor::full({2, 2}, 42.0f);
    // std::cout << "Tensor t5 (full with 42.0):" << std::endl;
    // t5.print();

    // Tensor t6 = t5 + t4;
    // t6.print();
    //
    // Tensor t7 = t5 * t4;
    // t7.print();
    //
    //
    // af::array a = af::randu(3, 4);       // shape: (3, 4)
    // af::array b = af::randu(3, 1);       // shape: (3, 1)
    //
    // af::array c = a + b;  // b is broadcast along 2nd dim
    //
    // af_print(a);
    // af_print(b);
    // af_print(c);
    Tensor a = Tensor::full({2, 2}, 2.0f, true);
    Tensor b = Tensor::full({2, 2}, 3.0f, true);

    // c = a + b
    Tensor c = a + b;

    // d = c * b
    Tensor d = c * b;

    // e = d + a
    Tensor e = d + a;

    e.backward();  // Start backward from e

    // Print results
    std::cout << "a:"; a.print();
    std::cout << "b:"; b.print();
    std::cout << "c:"; c.print();
    std::cout << "d:"; d.print();
    std::cout << "e:"; e.print();

    std::cout << "a.grad:"; af_print(a.grad());  // Expect 4.0
    std::cout << "b.grad:"; af_print(b.grad());  // Expect 8.0
    std::cout << "c.grad:"; af_print(c.grad());  // Intermediate
    std::cout << "d.grad:"; af_print(d.grad());  // Should be 1.0
    std::cout << "e.grad:"; af_print(e.grad());  // Should be 1.0

    return 0;
}