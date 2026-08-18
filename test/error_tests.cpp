#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>

#include <string_view>

TEST_CASE("vkexec error category maps errc values", "[vkexec][error]")
{
  REQUIRE(std::string_view(vkexec::category().Name()) == "vkexec");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::invalid_argument).Message() == "invalid argument");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::io_error).Message() == "I/O error");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::parse_error).Message() == "parse error");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::unsupported).Message() == "unsupported operation");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::out_of_range).Message() == "out of range");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::empty_result).Message() == "empty result");
  REQUIRE(vkexec::MakeErrorCode(static_cast<vkexec::errc>(999)).Message() == "unknown vkexec error");
}

TEST_CASE("make_error prefers detail over category message", "[vkexec][error]")
{
  vkexec::error const with_detail = vkexec::make_error(vkexec::errc::parse_error, "bad glsl");
  REQUIRE(with_detail.message() == "bad glsl");

  vkexec::error const without_detail = vkexec::make_error(vkexec::errc::parse_error);
  REQUIRE(without_detail.message() == "parse error");
}
