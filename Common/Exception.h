/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_UTIL_NPP_EXCEPTIONS_H
#define NV_UTIL_NPP_EXCEPTIONS_H

#include <exception>
#include <sstream>
#include <string>

/// All npp related C++ classes are put into the npp namespace.
namespace npp {

/// Exception base class.
///     This exception base class will be used for everything C++ throught
/// the NPP project.
///     The exception contains a string message, as well as data fields for a
///     string
/// containing the name of the file as well as the line number where the
/// exception was thrown.
///     The easiest way of throwing exceptions and providing filename and line
///     number is
/// to use one of the ASSERT macros defined for that purpose.
// class Exception
class Exception : public std::exception {
 public:
  /// Constructor.
  /// \param rMessage A message with information as to why the exception was
  /// thrown. \param rFileName The name of the file where the exception was
  /// thrown. \param nLineNumber Line number in the file where the exception was
  /// thrown.
  explicit Exception(const std::string &rMessage = "",
                     const std::string &rFileName = "",
                     unsigned int nLineNumber = 0)
      : std::exception(),
        sMessage_(rMessage),
        sFileName_(rFileName),
        nLineNumber_(nLineNumber),
        sWhatArg_(sFileName_ + ":" + std::to_string(nLineNumber_) + " : " +
                  sMessage_){

        };

  Exception(const Exception &rException)
      : std::exception(),
        sMessage_(rException.sMessage_),
        sFileName_(rException.sFileName_),
        nLineNumber_(rException.nLineNumber_),
        sWhatArg_(sFileName_ + ":" + std::to_string(nLineNumber_) + ": " +
                  sMessage_){

        };

  virtual ~Exception() = default;

  /// Get the exception's message.
  const std::string &message() const { return sMessage_; }

  /// Get the exception's file info.
  const std::string &fileName() const { return sFileName_; }

  /// Get the exceptions's line info.
  unsigned int lineNumber() const { return nLineNumber_; }

  /// Create a clone of this exception.
  ///      This creates a new Exception object on the heap. It is
  /// the responsibility of the user of this function to free this memory
  /// (delete x).
  virtual Exception *clone() const { return new Exception(*this); }

  /// Create a single string with all the exceptions information.
  ///     The virtual toString() method is used by the operator<<()
  /// so that all exceptions derived from this base-class can print
  /// their full information correctly even if a reference to their
  /// exact type is not had at the time of printing (i.e. the basic
  /// operator<<() is used).
  virtual std::string toString() const {
    // std::ostringstream oOutputString;
    // oOutputString << fileName() << ":" << lineNumber() << ": " << message();
    // return oOutputString.str();
    return sWhatArg_;
  }

  const char *what() const noexcept { return sWhatArg_.c_str(); }

 private:
  std::string sMessage_;   ///< Message regarding the cause of the exception.
  std::string sFileName_;  ///< Name of the file where the exception was thrown.
  unsigned int
      nLineNumber_;  ///< Line number in the file where the exception was thrown
  std::string sWhatArg_;
};

/// Output stream inserter for Exception.
/// \param rOutputStream The stream the exception information is written to.
/// \param rException The exception that's being written.
/// \return Reference to the output stream being used.
std::ostream &operator<<(std::ostream &rOutputStream,
                         const Exception &rException) {
  rOutputStream << rException.toString();
  return rOutputStream;
}

/// Basic assert macro.
///     This macro should be used to enforce any kind of pre or post conditions.
/// Unlike the C-runtime assert macro, this macro does not abort execution, but
/// throws a C++ exception. The exception is automatically filled with
/// information about the failing condition, the filename and line number where
/// the exception was thrown. \note The macro is written in such a way that
/// omitting a semicolon after its usage
///     causes a compiler error. The correct way to invoke this macro is:
/// NPP_ASSERT(n < MAX);
#define NPP_ASSERT(C)                                                   \
  do {                                                                  \
    if (!(C))                                                           \
      throw npp::Exception(#C " assertion faild!", __FILE__, __LINE__); \
  } while (false)

#define NPP_ENFORCE(C)                                                        \
  do {                                                                        \
    if (!(C)) throw npp::Exception(#C " enforce faild!", __FILE__, __LINE__); \
  } while (false)

// ASSERT macro.
//  Same functionality as the basic assert macro with the added ability to pass
//  a message M. M should be a string literal.
//  Note: Never use code inside ASSERT() that causes a side-effect ASSERT macros
//  may get compiled
//      out in release mode.
#define NPP_ASSERT_MSG(C, M)                                             \
  do {                                                                   \
    if (!(C))                                                            \
      throw npp::Exception(#C " assertion faild! Message: " M, __FILE__, \
                           __LINE__);                                    \
  } while (false)

#define NPP_ENFORCE_MSG(C, M)                                          \
  do {                                                                 \
    if (!(C))                                                          \
      throw npp::Exception(#C " enforce faild! Message: " M, __FILE__, \
                           __LINE__);                                  \
  } while (false)

#ifdef _DEBUG
/// Basic debug assert macro.
///     This macro is identical in every respect to NPP_ASSERT(C) but it does
///     get compiled to a
/// no-op in release builds. It is therefor of utmost importance to not put
/// statements into this macro that cause side effects required for correct
/// program execution.
#define NPP_DEBUG_ASSERT(C)                                                   \
  do {                                                                        \
    if (!(C))                                                                 \
      throw npp::Exception(#C " debug assertion faild!", __FILE__, __LINE__); \
  } while (false)
#else
#define NPP_DEBUG_ASSERT(C)
#endif

/// ASSERT for null-pointer test.
/// It is safe to put code with side effects into this macro. Also: This macro
/// never gets compiled to a no-op because resource allocation may fail based on
/// external causes not under control of a software developer.
#define NPP_ASSERT_NOT_NULL(P)                                        \
  do {                                                                \
    if ((P) == 0)                                                     \
      throw npp::Exception(#P " not null assertion faild!", __FILE__, \
                           __LINE__);                                 \
  } while (false)

/// Macro for flagging methods as not implemented.
/// The macro throws an exception with a message that an implementation was
/// missing
#define NPP_NOT_IMPLEMENTED()                                            \
  do {                                                                   \
    throw npp::Exception("Implementation missing!", __FILE__, __LINE__); \
  } while (false)

/// Macro for checking error return code of CUDA (runtime) calls.
/// This macro never gets disabled.
#define NPP_CHECK_CUDA(S)                   \
  do {                                      \
    cudaError_t eCudaResult;                \
    eCudaResult = S;                        \
    NPP_ASSERT(eCudaResult == cudaSuccess); \
  } while (false)

/// Macro for checking error return code for NPP calls.
#define NPP_CHECK_NPP(S)                   \
  do {                                     \
    NppStatus eStatusNPP;                  \
    eStatusNPP = S;                        \
    NPP_ASSERT(eStatusNPP == NPP_SUCCESS); \
  } while (false)

/// Macro for checking error return codes from cuFFT calls.
#define NPP_CHECK_CUFFT(S)                     \
  do {                                         \
    cufftResult eCuFftResult;                  \
    eCuFftResult = S;                          \
    NPP_ASSERT(eCuFftResult == CUFFT_SUCCESS); \
  } while (false)

}  // namespace npp

#endif  // NV_UTIL_NPP_EXCEPTIONS_H
