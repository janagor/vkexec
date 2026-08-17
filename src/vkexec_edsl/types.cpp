#include <vkexec_edsl/ast.hpp>
#include <vkexec_edsl/types.hpp>

#include <cstdint>
#include <utility>

namespace vkexec::edsl {
namespace {

  auto binary_op(OpKind kind, int lhs, int rhs, ValueType type = ValueType::Float) -> int
  {
    ExprNode node = ExprNode::make(kind, lhs, rhs);
    node.type = type;
    return ast().append(std::move(node));
  }

  auto unary_op(OpKind kind, int lhs, ValueType type = ValueType::Float) -> int
  {
    ExprNode node = ExprNode::make(kind, lhs);
    node.type = type;
    return ast().append(std::move(node));
  }

  auto emit_assign(int dst_var, int src) -> void { ast().append(ExprNode::make(OpKind::Assign, dst_var, src)); }

}// namespace

auto Bool::constant(bool value) -> Bool
{
  ExprNode node = ExprNode::make(OpKind::ConstBool);
  node.const_i = value ? 1 : 0;
  node.type = ValueType::Bool;
  return Bool{ ast().append(std::move(node)) };
}

auto Int::constant(std::int64_t value) -> Int
{
  ExprNode node = ExprNode::make(OpKind::ConstInt);
  node.const_i = value;
  node.type = ValueType::Int;
  return Int{ ast().append(std::move(node)) };
}

auto Int::param_index() -> Int
{
  ExprNode node = ExprNode::make(OpKind::ParamIndex);
  node.name = "idx";
  node.type = ValueType::Int;
  return Int{ ast().append(std::move(node)) };
}

auto Int::vertex_index() -> Int
{
  ExprNode node = ExprNode::make(OpKind::VertexIndex);
  node.name = "vid";
  node.type = ValueType::Int;
  return Int{ ast().append(std::move(node)) };
}

auto Float::constant(double value) -> Float
{
  ExprNode node = ExprNode::make(OpKind::ConstFloat);
  node.const_f = value;
  node.type = ValueType::Float;
  return Float{ ast().append(std::move(node)) };
}

auto operator+(Int lhs, Int rhs) -> Int { return Int{ binary_op(OpKind::Add, lhs.id, rhs.id, ValueType::Int) }; }
auto operator-(Int lhs, Int rhs) -> Int { return Int{ binary_op(OpKind::Sub, lhs.id, rhs.id, ValueType::Int) }; }
auto operator*(Int lhs, Int rhs) -> Int { return Int{ binary_op(OpKind::Mul, lhs.id, rhs.id, ValueType::Int) }; }
auto operator/(Int lhs, Int rhs) -> Int { return Int{ binary_op(OpKind::Div, lhs.id, rhs.id, ValueType::Int) }; }
auto operator-(Int lhs) -> Int { return Int{ unary_op(OpKind::Neg, lhs.id, ValueType::Int) }; }

auto operator+(Float lhs, Float rhs) -> Float { return Float{ binary_op(OpKind::Add, lhs.id, rhs.id) }; }
auto operator-(Float lhs, Float rhs) -> Float { return Float{ binary_op(OpKind::Sub, lhs.id, rhs.id) }; }
auto operator*(Float lhs, Float rhs) -> Float { return Float{ binary_op(OpKind::Mul, lhs.id, rhs.id) }; }
auto operator/(Float lhs, Float rhs) -> Float { return Float{ binary_op(OpKind::Div, lhs.id, rhs.id) }; }
auto operator-(Float lhs) -> Float { return Float{ unary_op(OpKind::Neg, lhs.id) }; }

auto operator+(Float lhs, double rhs) -> Float { return lhs + Float::constant(rhs); }
auto operator-(Float lhs, double rhs) -> Float { return lhs - Float::constant(rhs); }
auto operator*(Float lhs, double rhs) -> Float { return lhs * Float::constant(rhs); }
auto operator/(Float lhs, double rhs) -> Float { return lhs / Float::constant(rhs); }
auto operator+(double lhs, Float rhs) -> Float { return Float::constant(lhs) + rhs; }
auto operator-(double lhs, Float rhs) -> Float { return Float::constant(lhs) - rhs; }
auto operator*(double lhs, Float rhs) -> Float { return Float::constant(lhs) * rhs; }
auto operator/(double lhs, Float rhs) -> Float { return Float::constant(lhs) / rhs; }

auto operator<(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Less, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator<=(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LessEqual, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator>(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Greater, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator>=(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::GreaterEqual, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator==(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Equal, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator!=(Int lhs, Int rhs) -> Bool
{
  return Bool{ binary_op(OpKind::NotEqual, lhs.id, rhs.id, ValueType::Bool) };
}

auto operator<(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Less, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator<=(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LessEqual, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator>(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Greater, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator>=(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::GreaterEqual, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator==(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::Equal, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator!=(Float lhs, Float rhs) -> Bool
{
  return Bool{ binary_op(OpKind::NotEqual, lhs.id, rhs.id, ValueType::Bool) };
}

auto operator&&(Bool lhs, Bool rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LogicalAnd, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator||(Bool lhs, Bool rhs) -> Bool
{
  return Bool{ binary_op(OpKind::LogicalOr, lhs.id, rhs.id, ValueType::Bool) };
}
auto operator!(Bool lhs) -> Bool { return Bool{ unary_op(OpKind::LogicalNot, lhs.id, ValueType::Bool) }; }

auto sin(Float value) -> Float { return Float{ unary_op(OpKind::Sin, value.id) }; }
auto cos(Float value) -> Float { return Float{ unary_op(OpKind::Cos, value.id) }; }
auto sqrt(Float value) -> Float { return Float{ unary_op(OpKind::Sqrt, value.id) }; }
auto abs(Float value) -> Float { return Float{ unary_op(OpKind::Abs, value.id) }; }
auto floor(Float value) -> Float { return Float{ unary_op(OpKind::Floor, value.id) }; }
auto ceil(Float value) -> Float { return Float{ unary_op(OpKind::Ceil, value.id) }; }
auto min(Float lhs, Float rhs) -> Float { return Float{ binary_op(OpKind::Min, lhs.id, rhs.id) }; }
auto max(Float lhs, Float rhs) -> Float { return Float{ binary_op(OpKind::Max, lhs.id, rhs.id) }; }

auto select(Bool cond, Float when_true, Float when_false) -> Float
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Float;
  return Float{ ast().append(std::move(node)) };
}
auto select(Bool cond, Int when_true, Int when_false) -> Int
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Int;
  return Int{ ast().append(std::move(node)) };
}
auto select(Bool cond, Float2 when_true, Float2 when_false) -> Float2
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Vec2;
  return Float2{ ast().append(std::move(node)) };
}
auto select(Bool cond, Float3 when_true, Float3 when_false) -> Float3
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Vec3;
  return Float3{ ast().append(std::move(node)) };
}
auto select(Bool cond, Float4 when_true, Float4 when_false) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Select, cond.id, when_true.id, when_false.id);
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}

auto vec2(Float coord_x, Float coord_y) -> Float2
{
  ExprNode node = ExprNode::make(OpKind::Vec2, coord_x.id, coord_y.id);
  node.type = ValueType::Vec2;
  return Float2{ ast().append(std::move(node)) };
}
auto vec2(double coord_x, double coord_y) -> Float2 { return vec2(Float::constant(coord_x), Float::constant(coord_y)); }

auto vec3(Float coord_x, Float coord_y, Float coord_z) -> Float3
{
  ExprNode node = ExprNode::make(OpKind::Vec3, coord_x.id, coord_y.id, coord_z.id);
  node.type = ValueType::Vec3;
  return Float3{ ast().append(std::move(node)) };
}
auto vec3(double coord_x, double coord_y, double coord_z) -> Float3
{
  return vec3(Float::constant(coord_x), Float::constant(coord_y), Float::constant(coord_z));
}

auto vec4(Float coord_x, Float coord_y, Float coord_z, Float coord_w) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Vec4, coord_x.id, coord_y.id, coord_z.id, coord_w.id);
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}
auto vec4(Float2 vec, Float coord_z, Float coord_w) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Vec4From2, vec.id, coord_z.id, coord_w.id);
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}
auto vec4(Float3 vec, Float coord_w) -> Float4
{
  ExprNode node = ExprNode::make(OpKind::Vec4From2, vec.id, coord_w.id);// reuse: vec4(vec3, float)
  node.kind = OpKind::Vec4From2;
  node.name = "vec3";
  node.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(node)) };
}
auto vec4(Float3 vec, double coord_w) -> Float4 { return vec4(vec, Float::constant(coord_w)); }

auto Int::operator=(Int const &other) -> Int &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("i", ValueType::Int); }
  emit_assign(id, other.id);
  return *this;
}
// NOLINTNEXTLINE(bugprone-exception-escape,cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
auto Int::operator=(Int &&other) -> Int & { return (*this = other); }
auto Int::operator+=(Int other) -> Int & { return (*this = *this + other); }
auto Int::operator-=(Int other) -> Int & { return (*this = *this - other); }
auto Int::operator*=(Int other) -> Int & { return (*this = *this * other); }
auto Int::operator/=(Int other) -> Int & { return (*this = *this / other); }

auto Float::operator=(Float const &other) -> Float &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("f", ValueType::Float); }
  emit_assign(id, other.id);
  return *this;
}
// NOLINTNEXTLINE(bugprone-exception-escape,cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
auto Float::operator=(Float &&other) -> Float & { return (*this = other); }
auto Float::operator+=(Float other) -> Float & { return (*this = *this + other); }
auto Float::operator-=(Float other) -> Float & { return (*this = *this - other); }
auto Float::operator*=(Float other) -> Float & { return (*this = *this * other); }
auto Float::operator/=(Float other) -> Float & { return (*this = *this / other); }

auto Float2::operator=(Float2 const &other) -> Float2 &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("v2", ValueType::Vec2); }
  emit_assign(id, other.id);
  return *this;
}
// NOLINTNEXTLINE(bugprone-exception-escape,cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
auto Float2::operator=(Float2 &&other) -> Float2 & { return (*this = other); }
auto Float3::operator=(Float3 const &other) -> Float3 &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("v3", ValueType::Vec3); }
  emit_assign(id, other.id);
  return *this;
}
// NOLINTNEXTLINE(bugprone-exception-escape,cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
auto Float3::operator=(Float3 &&other) -> Float3 & { return (*this = other); }
auto Float4::operator=(Float4 const &other) -> Float4 &
{
  if (this == &other) { return *this; }
  if (id < 0) { id = ast().make_temp("v4", ValueType::Vec4); }
  emit_assign(id, other.id);
  return *this;
}
// NOLINTNEXTLINE(bugprone-exception-escape,cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
auto Float4::operator=(Float4 &&other) -> Float4 & { return (*this = other); }

auto VertexWriter::position(Float2 pos_xy) -> void
{
  Float4 const clip = vec4(pos_xy, Float::constant(0.0), Float::constant(1.0));
  position(clip);
}
auto VertexWriter::position(Float4 clip) -> void
{
  ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, clip.id);
  node.name = "gl_Position";
  node.type = ValueType::Vec4;
  ast().append(std::move(node));
}
auto VertexWriter::point_size(Float size) -> void
{
  ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, size.id);
  node.name = "gl_PointSize";
  node.type = ValueType::Float;
  ast().append(std::move(node));
}
auto VertexWriter::color(Float3 rgb) -> void
{
  ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, rgb.id);
  node.name = "vColor";
  node.type = ValueType::Vec3;
  node.const_i = 0;// location
  ast().append(std::move(node));
}
auto VertexWriter::color(Float4 rgba) -> void
{
  ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, rgba.id);
  node.name = "vColor4";
  node.type = ValueType::Vec4;
  node.const_i = 0;
  ast().append(std::move(node));
}

auto FragmentReader::color() -> Float3
{
  ExprNode node = ExprNode::make(OpKind::InputVarying);
  node.name = "vColor";
  node.type = ValueType::Vec3;
  node.const_i = 0;
  return Float3{ ast().append(std::move(node)) };
}
auto FragmentReader::color4() -> Float4
{
  ExprNode node = ExprNode::make(OpKind::InputVarying);
  node.name = "vColor4";
  node.type = ValueType::Vec4;
  node.const_i = 0;
  return Float4{ ast().append(std::move(node)) };
}

auto FragmentWriter::color(Float4 rgba) -> void
{
  ExprNode node = ExprNode::make(OpKind::OutputVarying, -1, rgba.id);
  node.name = "fragColor";
  node.type = ValueType::Vec4;
  node.const_i = 0;
  ast().append(std::move(node));
}

}// namespace vkexec::edsl
