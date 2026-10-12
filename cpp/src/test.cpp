// // create a dumb function in a class to test the bindings
// namespace Test {
//   int test_function(int x) {
//     return x * 2;
//   }
// };

#include <test.hpp>

int Test::test_function(int x) {
  return x * 2;
}