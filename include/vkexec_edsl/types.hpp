#ifndef VKEXEC_EDSL_TYPES_HPP
#define VKEXEC_EDSL_TYPES_HPP

#include <cstdint>

namespace vkexec::edsl {

struct Bool;
struct Int;
struct Float;
struct Float2;
struct Float3;
struct Float4;

struct Bool
{
  int id{ -1 };

  Bool() = default;
  explicit Bool(int node_id) : id(node_id) {}
  Bool(Bool const &) = default;
  auto operator=(Bool const &) -> Bool & = default;
  Bool(Bool &&) = default;
  auto operator=(Bool &&) -> Bool & = default;
  ~Bool() = default;

  static auto constant(bool value) -> Bool;
};

struct Int
{
  int id{ -1 };

  Int() = default;
  explicit Int(int node_id) : id(node_id) {}
  Int(Int const &) = default;
  auto operator=(Int const &other) -> Int &;
  Int(Int &&) = default;
  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
  auto operator=(Int &&other) -> Int &;
  ~Int() = default;
  auto operator+=(Int other) -> Int &;
  auto operator-=(Int other) -> Int &;
  auto operator*=(Int other) -> Int &;
  auto operator/=(Int other) -> Int &;

  static auto constant(std::int64_t value) -> Int;
  static auto param_index() -> Int;
  static auto vertex_index() -> Int;
};

struct Float
{
  int id{ -1 };

  Float() = default;
  explicit Float(int node_id) : id(node_id) {}
  Float(Float const &) = default;
  auto operator=(Float const &other) -> Float &;
  Float(Float &&) = default;
  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
  auto operator=(Float &&other) -> Float &;
  ~Float() = default;
  auto operator+=(Float other) -> Float &;
  auto operator-=(Float other) -> Float &;
  auto operator*=(Float other) -> Float &;
  auto operator/=(Float other) -> Float &;

  static auto constant(double value) -> Float;
};

struct Float2
{
  int id{ -1 };
  Float2() = default;
  explicit Float2(int node_id) : id(node_id) {}
  Float2(Float2 const &) = default;
  auto operator=(Float2 const &other) -> Float2 &;
  Float2(Float2 &&) = default;
  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
  auto operator=(Float2 &&other) -> Float2 &;
  ~Float2() = default;
};

struct Float3
{
  int id{ -1 };
  Float3() = default;
  explicit Float3(int node_id) : id(node_id) {}
  Float3(Float3 const &) = default;
  auto operator=(Float3 const &other) -> Float3 &;
  Float3(Float3 &&) = default;
  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
  auto operator=(Float3 &&other) -> Float3 &;
  ~Float3() = default;
};

struct Float4
{
  int id{ -1 };
  Float4() = default;
  explicit Float4(int node_id) : id(node_id) {}
  Float4(Float4 const &) = default;
  auto operator=(Float4 const &other) -> Float4 &;
  Float4(Float4 &&) = default;
  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
  auto operator=(Float4 &&other) -> Float4 &;
  ~Float4() = default;
};

auto operator+(Int lhs, Int rhs) -> Int;
auto operator-(Int lhs, Int rhs) -> Int;
auto operator*(Int lhs, Int rhs) -> Int;
auto operator/(Int lhs, Int rhs) -> Int;
auto operator-(Int lhs) -> Int;

auto operator+(Float lhs, Float rhs) -> Float;
auto operator-(Float lhs, Float rhs) -> Float;
auto operator*(Float lhs, Float rhs) -> Float;
auto operator/(Float lhs, Float rhs) -> Float;
auto operator-(Float lhs) -> Float;

auto operator+(Float lhs, double rhs) -> Float;
auto operator-(Float lhs, double rhs) -> Float;
auto operator*(Float lhs, double rhs) -> Float;
auto operator/(Float lhs, double rhs) -> Float;
auto operator+(double lhs, Float rhs) -> Float;
auto operator-(double lhs, Float rhs) -> Float;
auto operator*(double lhs, Float rhs) -> Float;
auto operator/(double lhs, Float rhs) -> Float;

auto operator<(Int lhs, Int rhs) -> Bool;
auto operator<=(Int lhs, Int rhs) -> Bool;
auto operator>(Int lhs, Int rhs) -> Bool;
auto operator>=(Int lhs, Int rhs) -> Bool;
auto operator==(Int lhs, Int rhs) -> Bool;
auto operator!=(Int lhs, Int rhs) -> Bool;

auto operator<(Float lhs, Float rhs) -> Bool;
auto operator<=(Float lhs, Float rhs) -> Bool;
auto operator>(Float lhs, Float rhs) -> Bool;
auto operator>=(Float lhs, Float rhs) -> Bool;
auto operator==(Float lhs, Float rhs) -> Bool;
auto operator!=(Float lhs, Float rhs) -> Bool;

auto operator&&(Bool lhs, Bool rhs) -> Bool;
auto operator||(Bool lhs, Bool rhs) -> Bool;
auto operator!(Bool lhs) -> Bool;

auto sin(Float value) -> Float;
auto cos(Float value) -> Float;
auto sqrt(Float value) -> Float;
auto abs(Float value) -> Float;
auto floor(Float value) -> Float;
auto ceil(Float value) -> Float;
auto min(Float lhs, Float rhs) -> Float;
auto max(Float lhs, Float rhs) -> Float;

auto select(Bool cond, Float when_true, Float when_false) -> Float;
auto select(Bool cond, Int when_true, Int when_false) -> Int;
auto select(Bool cond, Float2 when_true, Float2 when_false) -> Float2;
auto select(Bool cond, Float3 when_true, Float3 when_false) -> Float3;
auto select(Bool cond, Float4 when_true, Float4 when_false) -> Float4;

auto vec2(Float coord_x, Float coord_y) -> Float2;
auto vec2(double coord_x, double coord_y) -> Float2;
auto vec3(Float coord_x, Float coord_y, Float coord_z) -> Float3;
auto vec3(double coord_x, double coord_y, double coord_z) -> Float3;
auto vec4(Float coord_x, Float coord_y, Float coord_z, Float coord_w) -> Float4;
auto vec4(Float2 vec, Float coord_z, Float coord_w) -> Float4;
auto vec4(Float3 vec, Float coord_w) -> Float4;
auto vec4(Float3 vec, double coord_w) -> Float4;

/// Vertex-stage outputs written during tracing.
struct VertexWriter
{
  static auto position(Float2 pos_xy) -> void;
  static auto position(Float4 clip) -> void;
  static auto point_size(Float size) -> void;
  static auto color(Float3 rgb) -> void;
  static auto color(Float4 rgba) -> void;
};

/// Fragment-stage interpolated inputs.
struct FragmentReader
{
  [[nodiscard]] static auto color() -> Float3;
  [[nodiscard]] static auto color4() -> Float4;
};

/// Fragment-stage color attachment outputs.
struct FragmentWriter
{
  static auto color(Float4 rgba) -> void;
};

}// namespace vkexec::edsl

#endif// VKEXEC_EDSL_TYPES_HPP
