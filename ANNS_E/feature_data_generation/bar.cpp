#include <highfive/H5Easy.hpp>
#include <iostream>

int main() {
    H5Easy::File file("example.h5", H5Easy::File::Overwrite);

    int A = 123;
    H5Easy::dump(file, "/path/to/A", A);

    A = H5Easy::load<int>(file, "/path/to/A");
    std::cout << A << std::endl;
}