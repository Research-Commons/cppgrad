#include "ops/add.h"
#include <stdexcept>

namespace cppgrad {
    Tensor add(const Tensor& a, const Tensor& b) {
        //use for backwards later
    }

    Tensor operator+(const Tensor& a, const Tensor& b) {
        //will change this once broadcasting is implemented, for now it will throw and error if shape doesnt match
        if (a.shape() != b.shape())
            throw std::runtime_error("Shape mismatch in add");

        Tensor out(a.data_ + b.data_, a.requires_grad() || b.requires_grad());

        //use for backwards later

        return out;
    }
}
