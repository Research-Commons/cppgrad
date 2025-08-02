#include <fstream>

#include "tensor/tensor.hpp"
#include "tensor/tensorutils.hpp"
#include <iostream>

#include "autograd/graphviz_exporter.hpp"

using namespace cppgrad;

int main() {
    af::info();

    // Test: Manual construction
    std::vector<float> values = {1, 2, 3, 4, 5, 6};
    Tensor t1({2, 3}, values);
    std::cout << "Tensor t1 (manual):" << std::endl;
    t1.print();
    t1.print_pretty();

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

    std::cout << "\nTest 9: Throw debug only warning if backward called twice\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = Tensor::full({}, 2.0, true);

        Tensor c = a * b;

        c.backward();
        c.backward(); // this will throw [debug] backward() called more than once on the same tensor in debug mode

        //This is only a warning as we anyway set gradient to 1 during backward call
    }

    std::cout << "\nTest 10: Throw debug only warning if backward called twice\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        Tensor b = Tensor::full({}, 3.0, true);
        Tensor c = a * b;    // b = 6
        c.backward();
        a.print_grad(); // → prints 3

        b.backward();
        a.print_grad();
    }

    std::cout << "\nTest 11: Better print function\n";
    {
        Tensor a = Tensor::full({}, 2.0, true);
        a.print_pretty();
    }

    std::cout << "\nTest 12: Scalar Add\n";
    {
        Tensor a = Tensor::full({2, 1}, 2.0, true);

        Tensor b = a + 5.f;
        Tensor c = 5.f + b;

        c.backward(); // 1

        a.print_grad();

        b.print();
        c.print();
    }

    std::cout << "\nTest 13: Scalar Mul\n";
    {
        Tensor a = Tensor::full({2, 1}, 2.0, true);

        Tensor b = a * 5.f;
        Tensor c = 5.f * b;

        c.backward(); // 25

        a.print_grad();

        b.print();
        c.print();
    }

    std::cout << "Test 14: Clone Test:\n";
    {
        Tensor a = Tensor::full({2, 2}, 3.0f, true);

        // Just copy data, no gradient tracking
        Tensor b = TensorUtils::clone(a);

        a.print_pretty();
        b.print_pretty();

        //b.backward(); // this will throw an error cuz no autograd
    }

    std::cout << "\n\nTest 15: Clone Test With Autograd:\n";
    {
        // Original tensor `a`
        Tensor a = Tensor::full({2, 2}, 3.0f, true);  // requires_grad = true

        // Clone it
        Tensor b = TensorUtils::clone_with_grad(a);

        // Operate on clone
        Tensor c = b * 2.0f;

        // Backward pass
        c.backward();

        // Output values and gradients
        std::cout << "a:\n"; a.print();
        std::cout << "a.grad:\n"; a.print_grad();

        std::cout << "b:\n"; b.print();
        std::cout << "b.grad:\n"; b.print_grad();

        std::cout << "c:\n"; c.print();
        std::cout << "c.grad:\n"; c.print_grad();
    }

    std::cout << "\nTest 16: MatMul Forward (using full)\n";
    {
        // Initialize 'a' with shape 2×3 and values {1,2,3,4,5,6}
        std::vector<float> values_a = {1, 2, 3,
                                       4, 5, 6};
        Tensor a({2, 3}, values_a);  // shape: (2 × 3)

        // Initialize 'b' with shape 3×2 and values {7,8,9,10,11,12}
        std::vector<float> values_b = {7,  8,
                                       9, 10,
                                       11,12};
        Tensor b({3, 2}, values_b);  // shape: (3 × 2)

        // Perform matmul: c = a @ b → result shape (2 × 2)
        // Expected values:
        // [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12]  = [58,  64]
        // [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]  = [139,154]
        Tensor c = TensorUtils::matmul(a, b);

        // Print the result
        c.print();  // should output:
        // [[ 58,  64],
        //  [139, 154]]
    }

    std::cout << "\nTest 17: 4d tensor row major test\n";
    {
        // shape = 2×2×2, values 1…8 in row‑major:
        //  slice 0: [[1,2],[3,4]]
        //  slice 1: [[5,6],[7,8]]
        std::vector<float> vals3d = {
            1, 2,
            3, 4,

            5, 6,
            7, 8,

            9, 10,
            11, 12,

            13, 14,
            15, 16
        };
        Tensor t3({2,2,2, 2}, vals3d, /*requires_grad=*/false);

        // after constructing t3...
        auto A = t3.impl()->data();
        std::vector<float> host(vals3d.size());
        A.host(host.data());    // copy back the raw buffer

        // Now print the host vector in flat order:
        std::cout << "host = { ";
        for (size_t i = 0; i < host.size(); ++i) {
            std::cout << host[i];
            if (i+1 < host.size()) std::cout << ", ";
        }
        std::cout << " }\n";

        t3.print();
        t3.print_pretty();
    }

    std::cout << "\nTest 18: tensor column major test\n";
    {
        Tensor t = Tensor::from_array_column_major({2, 3}, {1, 2, 3, 4, 5, 6});

        t.print_pretty();

    }

    std::cout << "\nTest 19: new ops test\n";
    {
        Tensor a = Tensor::full({2, 2}, 3.0f, true);  // requires_grad = true
        Tensor b = Tensor::full({2, 2}, 2.0f, true);  // requires_grad = true

        // Expression: ((a + b) * (a - b)) / (a * b)
        Tensor c = a + b;    // c = a + b
        Tensor d = a - b;    // d = a - b
        Tensor e = c * d;    // e = (a + b) * (a - b)
        Tensor f = a * b;    // f = a * b
        Tensor out = e / f;  // out = ((a + b)*(a - b)) / (a * b)

        out.backward();

        std::cout << "\nGradient wrt a:\n";a.print_grad();
        std::cout << "\nGradient wrt b:\n";b.print_grad();

//         a.grad() = [[0.7222, 0.7222],
//             [0.7222, 0.7222]]
//
//         b.grad() = [[-1.0833, -1.0833],
//             [-1.0833, -1.0833]]

        //Visualizer::save_dot(out, "graph");

    }

    std::cout << "\nTest 20: new new ops test\n";
    {
        Tensor a = Tensor::full({2, 2}, 3.0f, true);  // requires_grad = true
        Tensor b = Tensor::full({2, 2}, 2.0f, true);  // requires_grad = true

        // Core expression: ((a + b) * (a - b)) / (a * b)
        Tensor c = a + b;
        Tensor d = a - b;
        Tensor e = c * d;
        Tensor f = a * b;
        Tensor frac = e / f;

        // New: log(frac) + exp(-a) + pow(b, a)
        Tensor log_part = log(frac);
        Tensor neg_a = -a;
        Tensor exp_part = exp(neg_a);
        Tensor pow_part = pow(b, a);

        Tensor out = log_part + exp_part + pow_part;

        // Backprop
        out.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();
        std::cout << "\nGradient wrt b:\n";
        b.print_grad();

        // a.grad	≈ 6.362
        // b.grad	≈ 10.7

        //Visualizer::save_dot(out, "graph");

    }

    std::cout << "\nTest 21: sum over all elements\n";
    {
        Tensor a = Tensor::full({2, 2}, 1.0f, true);  // [[1, 1], [1, 1]]
        Tensor s = a.sum();                          // scalar: 4.0

        s.print();  // Should print scalar 4.0
        s.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Should be [[1, 1], [1, 1]]
    }

    std::cout << "\nTest 22: sum along dim=0, keepdim=false\n";
    {
        Tensor a ({2, 2}, {1 ,2 , 3, 4}, true);  // [[1, 2], [3, 4]]
        Tensor s = a.sum(0);                           // sum over rows → [4, 6]

        s.print();  // Should print shape (2,) with values [4, 6]
        s.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Should be [[1, 1], [1, 1]]
    }

    std::cout << "\nTest 23: sum along dim=1, keepdim=true\n";
    {
        Tensor a ({2, 2}, {1 ,2 , 3, 4}, true);  // [[1, 2], [3, 4]]
        Tensor s = a.sum(1, true);                     // sum across cols, keepdim → [[3], [7]]

        s.print();  // Should print shape (2,1) with values [[3], [7]]
        s.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Should be [[1, 1], [1, 1]]
    }

    std::cout << "\nTest 24: sum along dim=1 without keepdim\n";
    {
        Tensor a ({2, 3}, {0, 1 , 2, 3, 4, 5}, true);  // [[0, 1, 2], [3, 4, 5]]

        Tensor s = a.sum(1);  // Sum along dim=1 → shape: [2]

        Tensor out = s * Tensor::full({2}, 2.0f); // Multiply s by 2.0 → gradient will be 2.0

        out.backward();

        std::cout << "a:\n"; a.print();
        std::cout << "a.grad:\n"; a.print_grad();

        // Expected a.grad:
        // a.grad =
        // [[2, 2, 2],
        //  [2, 2, 2]]
    }

    std::cout << "\nTest 25: mean over all elements\n";
    {
        Tensor a = Tensor::full({2, 2}, 1.0f, true);  // [[1, 1], [1, 1]]
        Tensor m = a.mean();                         // scalar: 1.0

        m.print();  // Should print scalar 1.0
        m.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Should be [[0.25, 0.25], [0.25, 0.25]]
    }

    std::cout << "\nTest 26: mean along dim=0, keepdim=false\n";
    {
        Tensor a({2, 2}, {1, 2, 3, 4}, true);  // [[1, 2], [3, 4]]
        Tensor m = a.mean(0);                 // Mean over rows → [2.0, 3.0]

        m.print();  // Should print shape (2,) with values [2.0, 3.0]
        m.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Should be [[0.5, 0.5], [0.5, 0.5]]
    }

    std::cout << "\nTest 27: mean along dim=1, keepdim=true\n";
    {
        Tensor a({2, 2}, {1, 2, 3, 4}, true);  // [[1, 2], [3, 4]]
        Tensor m = a.mean(1, true);           // Mean across cols → [[1.5], [3.5]]

        m.print();  // Should print shape (2,1) with values [[1.5], [3.5]]
        m.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Should be [[0.5, 0.5], [0.5, 0.5]]
    }

    std::cout << "\nTest 28: mean along dim=1 without keepdim + scale\n";
    {
        Tensor a({2, 3}, {0, 1, 2, 3, 4, 5}, true);  // [[0, 1, 2], [3, 4, 5]]
        Tensor m = a.mean(1);                        // Mean along dim=1 → shape: [2]

        Tensor out = m * Tensor::full({2}, 3.0f);    // Multiply result by 3.0

        out.backward();

        std::cout << "a:\n"; a.print();
        std::cout << "a.grad:\n"; a.print_grad();

        // Expected a.grad:
        // [[1, 1, 1],     // 3.0 * d(mean)/dx = 3.0 * (1/3)
        //  [1, 1, 1]]     // Each row has gradient 1.0 broadcasted
    }

    std::cout << "\nTest 29: max over all elements\n";
    {
        Tensor a({2, 2}, {1, 10, 1, 1}, true);  // [[1, 10], [1, 1]]
        a.print();
        Tensor m = a.max();  // scalar 10.0
        m.print();
        m.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Only the max location should have 1.0, others 0
    }

    std::cout << "\nTest 30: max along dim=0, keepdim=false\n";
    {
        Tensor a({2, 2}, {1, 5, 3, 4}, true);  // [[1, 5], [3, 4]]
        Tensor m = a.max(0);  // max over rows → [3, 5]
        m.print();

        m.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Gradient only on [1][0] and [0][1]
    }

    std::cout << "\nTest 31: max along dim=1, keepdim=true\n";
    {
        Tensor a({2, 3}, {1, 9, 5, 2, 3, 6}, true);  // [[1, 9, 5], [2, 3, 6]]
        Tensor m = a.max(1, true);  // keepdim → shape (2,1)

        m.print();  // [[9], [6]]
        m.backward();

        std::cout << "\nGradient wrt a:\n";
        a.print_grad();  // Only max locations get gradient 1.0
        // Expected:
        // [[0, 1, 0],
        //  [0, 0, 1]]
    }

    std::cout << "\nTest 32: max(dim=1) followed by multiply\n";
    {
        Tensor a({2, 3}, {2, 4, 6, 1, 8, 7}, true);  // [[2,4,6], [1,8,7]]
        Tensor m = a.max(1);  // → [6, 8]

        Tensor out = m * Tensor::full({2}, 2.0f);  // → [12, 16]
        out.backward();

        std::cout << "a:\n"; a.print();
        std::cout << "a.grad:\n"; a.print_grad();

        // Expected grad:
        // [[0, 0, 2],
        //  [0, 2, 0]]
    }


    return 0;
}