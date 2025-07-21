#include "tensor/tensor.hpp"
#include <iostream>

using namespace cppgrad;

int main() {
    af::info();

    // Test: Manual construction
    std::vector<float> values = {1, 2, 3, 4, 5, 6};
    Tensor t1({2, 3}, values);
    std::cout << "Tensor t1 (manual):" << std::endl;
    t1.print();

    std::cout << "Shape: ";
    for (auto s : t1.shape()) std::cout << s << " ";

    std::cout << "\nNumel: " << t1.numel() << ", Dims: " << t1.ndim() << std::endl;
    std::cout << std::endl;

    // Test: zeros
    Tensor t2 = Tensor::zeros({2, 2});
    std::cout << "Tensor t2 (zeros):" << std::endl;
    t2.print();

    // Test: ones
    Tensor t3 = Tensor::ones({2, 3});
    std::cout << "Tensor t3 (ones):" << std::endl;
    t3.print();

    // Test: randn
    Tensor t4 = Tensor::randn({2, 2});
    std::cout << "Tensor t4 (randn):" << std::endl;
    t4.print();

    // Test: full
    Tensor t5 = Tensor::full({2, 2}, 42.0f);
    std::cout << "Tensor t5 (full with 42.0):" << std::endl;
    t5.print();

    Tensor t6 = t5 + t4;
    t6.print();

    Tensor t7 = t5 * t4;
    t7.print();


    std::cout << "Test 1: e = a*b + d\n";
    {
        Tensor a = Tensor::full({2, 2}, 3.0, true);
        Tensor b = Tensor::full({2, 2}, 4.0, true);
        Tensor d = Tensor::full({2, 2}, 2.0, true);
        Tensor c = a * b;
        Tensor e = c + d;
        e.backward();
        a.print_grad(); // 4.0
        b.print_grad(); // 3.0
        d.print_grad(); // 1.0
        c.print_grad(); // 1.0
        e.print_grad(); // 1.0
    }

    std::cout << "\nTest 2: z = a * b * c\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = Tensor::full({}, 3.0, true);
        Tensor c = Tensor::full({}, 4.0, true);
        Tensor z = a * b * c;
        z.backward();
        a.print_grad(); // 12.0
        b.print_grad(); // 8.0
        c.print_grad(); // 6.0
    }

    std::cout << "\nTest 3: p = (a + b) * b\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = Tensor::full({}, 3.0, true);
        Tensor s = a + b;
        Tensor p = s * b;
        p.backward();
        a.print_grad();  // 3.0
        b.print_grad(); // 8.0
    }

    std::cout << "\nTest 4: grads before backward\n";
    {
        Tensor a = Tensor::full({}, 5.0, true);
        Tensor b = Tensor::full({}, 7.0, true);
        Tensor z = a + b;
        a.print_grad(); // 0.0
        b.print_grad(); // 0.0
        z.backward();
        a.print_grad();  // 1.0
        b.print_grad(); // 1.0
    }

    std::cout << "\nTest 5: reuse x in multiple ops\n";
    {
        Tensor x = Tensor::full({}, 2.0, true);
        Tensor y1 = x * x;
        Tensor y2 = x + x;
        Tensor z = y1 + y2;
        z.backward();
        x.print_grad(); // 6.0
    }

    std::cout << "\nTest 6: constant tensor\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = Tensor::full({}, 3.0, false);
        Tensor c = a * b;
        c.backward();
        a.print_grad();  // 3.0
        b.print_grad();// b.grad() should not exist
    }

    std::cout << "\nTest 7: intermediate reuse\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = a * a;
        Tensor c = b * a;
        c.backward();
        a.print_grad();  // 12.0
    }

    std::cout << "\nTest 8: direct definition\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = a * a;
        Tensor c = b * Tensor::full({}, 5.0, true);
        c.backward();
        a.print_grad();  // 20.0
    }

    return 0;
}