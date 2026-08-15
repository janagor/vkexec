#ifndef VKEXEC_DETAIL_TYPES_HPP
#define VKEXEC_DETAIL_TYPES_HPP


#include <vkexec/detail/ast.hpp>

#include <cstdint>

namespace vlk {

struct Bool;
struct Int;
struct Float;
struct Float2;
struct Float3;
struct Float4;

struct Bool {
  int id{ -1 };

  Bool() = default;
  explicit Bool(int node_id) : id(node_id) {}
  Bool(const Bool &) = default;
  auto operator=(const Bool &) -> Bool & = default;
  Bool(Bool &&) = default;
  auto operator=(Bool &&) -> Bool & = default;
  ~Bool() = default;

  static auto constant(bool value) -> Bool
  {
    ExprNode node = ExprNode::make(OpKind::ConstBool);
    node.const_i = value ? 1 : 0;
    node.type = ValueType::Bool;
    return Bool{ ast().append(std::move(node)) };
  }
};

struct Int {
  int id{ -1 };

  Int() = default;
  explicit Int(int node_id) : id(node_id) {}
  Int(const Int &) = default;
  auto operator=(const Int &other) -> Int &;
  Int(Int &&) = default;
  auto operator=(Int &&) -> Int & = default;
  ~Int() = default;
  auto operator+=(Int other) -> Int &;
  auto operator-=(Int other) -> Int &;
  auto operator*=(Int other) -> Int &;
  auto operator/=(Int other) -> Int &;

  static auto constant(std::int64_t value) -> Int
  {
    ExprNode node = ExprNode::make(OpKind::ConstInt);
    node.const_i = value;
    node.type = ValueType::Int;
    return Int{ ast().append(std::move(node)) };
  }

  static auto param_index() -> Int
  {
    ExprNode node = ExprNode::make(OpKind::ParamIndex);
    node.name = "idx";
    node.type = ValueType::Int;
    return Int{ ast().append(std::move(node)) };
  }

  static auto vertex_index() -> Int
  {
    ExprNode node = ExprNode::make(OpKind::VertexIndex);
    node.name = "vid";
    node.type = ValueType::Int;
    return Int{ ast().append(std::move(node)) };
  }
};

struct Float {
  int id{ -1 };

  Float() = default;
  explicit Float(int node_id) : id(node_id) {}
  Float(const Float &) = default;
  auto operator=(const Float &other) -> Float &;
  Float(Float &&) = default;
  auto operator=(Float &&) -> Float & = default;
  ~Float() = default;
  auto operator+=(Float other) -> Float &;
  auto operator-=(Float other) -> Float &;
  auto operator*=(Float other) -> Float &;
  auto operator/=(Float other) -> Float &;

  static auto constant(double value) -> Float
  {
    ExprNode node = ExprNode::make(OpKind::ConstFloat);
    node.const_f = value;
    node.type = ValueType::Float;
    return Float{ ast().append(std::move(node)) };
  }
};

struct Float2 {
  int id{ -1 };
  Float2() = default;
  explicit Float2(int node_id) : id(node_id) {}
  Float2(const Float2 &) = default;
  auto operator=(const Float2 &other) -> Float2 &;
  Float2(Float2 &&) = default;
  auto operator=(Float2 &&) -> Float2 & = default;
  ~Float2() = default;
};

struct Float3 {
  int id{ -1 };
  Float3() = default;
  explicit Float3(int node_id) : id(node_id) {}
  Float3(const Float3 &) = default;
  auto operator=(const Float3 &other) -> Float3 &;
  Float3(Float3 &&) = default;
  auto operator=(Float3 &&) -> Float3 & = default;
  ~Float3() = default;
};

struct Float4 {
  int id{ -1 };
  Float4() = default;
  explicit Float4(int node_id) : id(node_id) {}
  Float4(const Float4 &) = default;
  auto operator=(const Float4 &other) -> Float4 &;
  Float4(Float4 &&) = default;
  auto operator=(Float4 &&) -> Float4 & = default;
  ~Float4() = default;
};

inline auto binary_op(OpKind kind, int lhs, int rhs, ValueType type = ValueType::Float) -> int
{
  ExprNode node = ExprNode::make(kind, lhs, rhs);
  node.type = type;
  return ast().append(std::move(node));
}

inline auto unary_op(OpKind kind, int lhs, ValueType type = ValueType::Float) -> int
{
  ExprNode node = ExprNode::make(kind, lhs);
  node.type = type;
  return ast().append(std::move(node));
}

inline auto operator+(Int lhs, Int rhs) -> Int
{
  return Int{ binary_op(OpKind::Add, lhs.id, rhs.id, ValueType::Int) };
}
inline auto operator-(Int lhs, Int rhs) -> Int
{
  return Int{ binary_op(OpKind::Sub, lhs.id, rhs.id, ValueType::Int) };
}
inline auto operator*(Int lhs, Int rhs) -> Int
{
  return Int{ binary_op(OpKind::Mul, lhs.id, rhs.id, ValueType::Int) };
}
inline auto operator/(Int lhs, Int rhs) -> Int
{
  return Int{ binary_op(OpKind::Div, lhs.id, rhs.id, ValueType::Int) };
}
inline auto operator-(Int lhs) -> Int { return Int{ unary_op(OpKind::Neg, lhs.id, ValueType::Int) }; }

inline auto operator+(Float lhs, Float rhs) -> Float
{
  return Float{ binary_op(OpKind::Add, lhs.id, rhs.id) };
}
inline auto operator-(Float lhs, Float rhs) -> Float
{
  return Float{ binary_op(OpKind::Sub, lhs.id, rhs.id) };
}
inline auto operator*(Float lhs, Float rhs) -> Float
{
  return Float{ binary_op(OpKind::Mul, lhs.id, rhs.id) };
}
inline auto operator/(Float lhs, Float rhs) -> Float
{
  return Float{ binary_op(OpKind::Div, lhs.id, rhs.id) };
}
inline auto operator-(Float lhs) -> Float { return Float{ unary_op(OpKind::Neg, lhs.id) }; }

inline auto operator+(Float lhs, double rhs) -> Float { return lhs + Float::constant(rhs); }
inline auto operator-(Float lhs, double rhs) -> Float { return lhs - Float::constant(rhs); }
inline auto operator*(Float lhs, double rhs) -> Float { return lhs * Float::constant(rhs); }
inline auto operator/(Float lhs, double rhs) -> Float { return lhs / Float::constant(rhs); }
inline auto operator+(double lhs, Float rhs) -> Float { return Float::constant(lhs) + rhs; }
inline auto operator-(double lhs, Float rhs) -> Float { return Float::constant(lhs) - rhs; }
inline auto operator*(double lhs, Float rhs) -> Float { return Float::constant(lhs) * rhs; }
inline auto operator/(double lhs, Float rhs) -> Float { return Float::constant(lhs) / rhs; }

inline auto operator<(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Less, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator<=(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LessEqual, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator>(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Greater, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator>=(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::GreaterEqual, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator==(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Equal, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator!=(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::NotEqual, lhs.id, rhs.id, ValueType::Bool) };
}

inline auto operator<(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Less, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator<=(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LessEqual, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator>(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Greater, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator>=(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::GreaterEqual, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator==(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Equal, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator!=(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::NotEqual, lhs.id, rhs.id, ValueType::Bool) };
}

inline auto operator&&(Bool lhs, Bool rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LogicalAnd, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator||(Bool lhs, Bool rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LogicalOr, lhs.id, rhs.id, ValueType::Bool) };
}
inline auto operator!(Bool lhs) -> Bool
{
  return Bool{ unary_op(OpKind::LogicalNot, lhs.id, ValueType::Bool) };
}

inline auto sin(Float value) -> Float { return Float{ unary_op(OpKind::Sin, value.id) }; }
inline auto cos(Float value) -> Float { return Float{ unary_op(OpKind::Cos, value.id) }; }
inline auto sqrt(Float value) -> Float { return Float{ unary_op(OpKind::Sqrt, value.id) }; }
inline auto abs(Float value) -> Float { return Float{ unary_op(OpKind::Abs, value.id) }; }
inline auto floor(Float value) -> Float { return Float{ unary_op(OpKind::Floor, value.id) }; }
inline auto ceil(Float value) -> Float { return Float{ unary_op(OpKind::Ceil, value.id) }; }
inline auto min(Float lhs, Float rhs) -> Float
{
  return Float{ binary_op(OpKind::Min, lhs.id, rhs.id) };
}
inline auto max(Float lhs, Float rhs) -> Float
{
  return Float{ binary_op(OpKind::Max, lhs.id, rhs.id) };
}

inline auto select(Bool cond, Float when_true, Float when_false) -> Float
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Float;
  return Float{ ast().append(std::move(node)) };
}
inline auto select(Bool cond, Int when_true, Int when_false) -> Int
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Int;
  return Int{ ast().append(std::move(node)) };
}
inline auto select(Bool cond, Float2 when_true, Float2 when_false) -> Float2
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Vec2;
  return Float2{ ast().append(std::move(node)) };
}
inline auto select(Bool cond, Float3 when_true, Float3 when_false) -> Float3
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Vec3;
  return Float3{ ast().append(std::move(node)) };
}
inline auto select(Bool cond, Float4 when_true, Float4 when_false) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}

inline auto vec2(Float coord_x, Float coord_y) -> Float2
{
  ExprNode node = ExprNode::make(OpKind::Vec2, coord_x.id, coord_y.id);
  node.type = ValueType::Vec2;
  return Float2{ ast().append(std::move(node)) };
}
inline auto vec2(double coord_x, double coord_y) -> Float2
{
  return vec2(Float::constant(coord_x), Float::constant(coord_y));
}

inline auto vec3(Float coord_x, Float coord_y, Float coord_z) -> Float3
{
  ExprNode node = ExprNode::make(OpKind::Vec3, coord_x.id, coord_y.id, coord_z.id);
  node.type = ValueType::Vec3;
  return Float3{ ast().append(std::move(node)) };
}
inline auto vec3(double coord_x, double coord_y, double coord_z) -> Float3
{
  return vec3(Float::constant(coord_x), Float::constant(coord_y), Float::constant(coord_z));
}

inline auto vec4(Float coord_x, Float coord_y, Float coord_z, Float coord_w) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Vec4, coord_x.id, coord_y.id, coord_z.id, coord_w.id);
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}
inline auto vec4(Float2 vec, Float coord_z, Float coord_w) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Vec4From2, vec.id, coord_z.id, coord_w.id);
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}
inline auto vec4(Float3 vec, Float coord_w) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Vec4From2, vec.id, coord_w.id); // reuse: vec4(vec3, float)
  node.kind = OpKind::Vec4From2;
  node.name = "vec3";
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}
inline auto vec4(Float3 vec, double coord_w) -> Float4 { return vec4(vec, Float::constant(coord_w)); }

inline auto emit_assign(int dst_var, int src) -> void
{
  ast().append(ExprNode::make(OpKind::Assign, dst_var, src));
}

inline auto Int::operator=(const Int &other) -> Int &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("i", ValueType::Int); }
  emit_assign(id, other.id);
  return *this;
}
inline auto Int::operator+=(Int other) -> Int & { return (*this = *this + other); }
inline auto Int::operator-=(Int other) -> Int & { return (*this = *this - other); }
inline auto Int::operator*=(Int other) -> Int & { return (*this = *this * other); }
inline auto Int::operator/=(Int other) -> Int & { return (*this = *this / other); }

inline auto Float::operator=(const Float &other) -> Float &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("f", ValueType::Float); }
  emit_assign(id, other.id);
  return *this;
}
inline auto Float::operator+=(Float other) -> Float & { return (*this = *this + other); }
inline auto Float::operator-=(Float other) -> Float & { return (*this = *this - other); }
inline auto Float::operator*=(Float other) -> Float & { return (*this = *this * other); }
inline auto Float::operator/=(Float other) -> Float & { return (*this = *this / other); }

inline auto Float2::operator=(const Float2 &other) -> Float2 &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("v2", ValueType::Vec2); }
  emit_assign(id, other.id);
  return *this;
}
inline auto Float3::operator=(const Float3 &other) -> Float3 &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("v3", ValueType::Vec3); }
  emit_assign(id, other.id);
  return *this;
}
inline auto Float4::operator=(const Float4 &other) -> Float4 &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("v4", ValueType::Vec4); }
  emit_assign(id, other.id);
  return *this;
}

/// Vertex-stage outputs written during tracing.
struct VertexWriter {
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto position(Float2 pos_xy) -> void
  {
    const Float4 clip = vec4(pos_xy, Float::constant(0.0), Float::constant(1.0));
    ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, clip.id);
    node.name = "gl_Position";
    node.type = ValueType::Vec4;
    ast().append(std::move(node));
  }
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto position(Float4 clip) -> void
  {
    ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, clip.id);
    node.name = "gl_Position";
    node.type = ValueType::Vec4;
    ast().append(std::move(node));
  }
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto point_size(Float size) -> void
  {
    ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, size.id);
    node.name = "gl_PointSize";
    node.type = ValueType::Float;
    ast().append(std::move(node));
  }
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto color(Float3 rgb) -> void
  {
    ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, rgb.id);
    node.name = "vColor";
    node.type = ValueType::Vec3;
    node.const_i = 0; // location
    ast().append(std::move(node));
  }
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto color(Float4 rgba) -> void
  {
    ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, rgba.id);
    node.name = "vColor4";
    node.type = ValueType::Vec4;
    node.const_i = 0;
    ast().append(std::move(node));
  }
};

/// Fragment-stage interpolated inputs.
struct FragmentReader {
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] auto color() const -> Float3
  {
    ExprNode node = ExprNode::make(OpKind::InputVarying);
    node.name = "vColor";
    node.type = ValueType::Vec3;
    node.const_i = 0;
    return Float3{ ast().append(std::move(node)) };
  }
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] auto color4() const -> Float4
  {
    ExprNode node = ExprNode::make(OpKind::InputVarying);
    node.name = "vColor4";
    node.type = ValueType::Vec4;
    node.const_i = 0;
    return Float4{ ast().append(std::move(node)) };
  }
};

/// Fragment-stage color attachment outputs.
struct FragmentWriter {
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto color(Float4 rgba) -> void
  {
    ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, rgba.id);
    node.name = "fragColor";
    node.type = ValueType::Vec4;
    node.const_i = 0;
    ast().append(std::move(node));
  }
};

} // namespace vlk

#endif  // VKEXEC_DETAIL_TYPES_HPP
