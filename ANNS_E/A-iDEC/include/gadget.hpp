
#ifndef _GADGET_HPP_
#define _GADGET_HPP_

#include <chrono>
namespace ss::gadget {

/// \brief SimpleProgressDisplay
///
///  A simple progress bae display!
///
///  !!! Shamelessly copy some codes from boost library !!!
///
struct SimpleProgressDisplay {
  explicit SimpleProgressDisplay(size_t n) :
      _tot(n),
      _display_steps((n + N_DISPLAYS - 1) / N_DISPLAYS),
      _cnt(0u) {}
  SimpleProgressDisplay &operator++() {
    if (_cnt == 0) {
      /// Copy from https://www.boost.org/doc/libs/1_57_0/boost/progress.hpp
      std::cout << "\n\n\t" << "0%   10   20   30   40   50   60   70   80   90   100%\n"
                << "\t" << "|----|----|----|----|----|----|----|----|----|----|"
                << std::endl;  // endl implies flush, which ensures display
    }
    ++_cnt;
    if (_cnt % _display_steps == 0) {
      if (_cnt == _display_steps) std::cout << "\t";
      std::cout << "*";
    }
    if (_cnt == _tot) {
      std::cout << "\n" << std::endl;
    }
  }
 private:
  size_t _tot;
  size_t _display_steps;
  size_t _cnt;
  static constexpr size_t N_DISPLAYS = 100u;
};

/// \brief SimpleTimer
///
using namespace std::chrono;
struct SimplerTimer {
  void restart() {
    _start = system_clock::now();
  }
  double elapsed() {
    auto cur = system_clock::now();
    return duration_cast<std::chrono::seconds>(cur - _start).count();
  }
 private:
  system_clock::time_point _start;
};

}// end namespace ss:gadget
#endif //_GADGET_HPP_
