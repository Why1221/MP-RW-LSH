#include <iostream>
#include <type_traits>
#include <cstdint>
#include <cmath>

void print_separator()
{
    std::cout << "-----\n";
}

int main()
{
    std::cout << std::boolalpha;

    // some implementation-defined facts
    std::cout << std::is_same<int, std::int32_t>::value << '\n';

    std::cout << std::is_same<int, const int>::value << '\n';
    // usually true if 'int' is 32 bit
    std::cout << std::is_same<int, std::int64_t>::value << '\n';
    // possibly true if ILP64 data model is used

    print_separator();

    // 'float' is never an integral type
    std::cout << std::is_same<float, std::int32_t>::value << '\n'; // false

    print_separator();

    // 'int' is implicitly 'signed'
    std::cout << std::is_same<int, int>::value << "\n";          // true
    std::cout << std::is_same<int, unsigned int>::value << "\n"; // false
    std::cout << std::is_same<int, signed int>::value << "\n";   // true

    print_separator();

    // unlike other types, 'char' is neither 'unsigned' nor 'signed'
    std::cout << std::is_same<char, char>::value << "\n";          // true
    std::cout << std::is_same<char, unsigned char>::value << "\n"; // false
    std::cout << std::is_same<char, signed char>::value << "\n";   // false
    std::cout << std::is_same<unsigned char, std::uint8_t >::value << "\n";          // true
    std::cout << std::is_same<bool, std::uint8_t >::value << "\n";          // false
    std::cout << std::is_same<bool, std::int8_t >::value << "\n";          // false

    std::cout << std::is_floating_point<const double>::value << '\n';

    uint32_t a = 6;
    uint32_t b = 8;
    int c = a - b;
    uint32_t res = std::abs(c);
    std::cout << "expected: 2" << ", got: " << res << "\n";
}
