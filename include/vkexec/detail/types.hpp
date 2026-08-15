#pragma once

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
  Bool &operator=(const Bool &) = default;

  static Bool constant(bool value)
  {
    ExprNode n = ExprNode::make(OpKind::ConstBool);
    n.const_i = value ? 1 : 0;
    n.type = ValueType::Bool;
    return Bool{ ast().append(std::move(n)) };
  }
};

struct Int {
  int id{ -1 };

  Int() = default;
  explicit Int(int node_id) : id(node_id) {}
  Int(const Int &) = default;
  Int &operator=(const Int &other);
  Int &operator+=(Int other);
  Int &operator-=(Int other);
  Int &operator*=(Int other);
  Int &operator/=(Int other);

  static Int constant(std::int64_t value)
  {
    ExprNode n = ExprNode::make(OpKind::ConstInt);
    n.const_i = value;
    n.type = ValueType::Int;
    return Int{ ast().append(std::move(n)) };
  }

  static Int param_index()
  {
    ExprNode n = ExprNode::make(OpKind::ParamIndex);
    n.name = "idx";
    n.type = ValueType::Int;
    return Int{ ast().append(std::move(n)) };
  }

  static Int vertex_index()
  {
    ExprNode n = ExprNode::make(OpKind::VertexIndex);
    n.name = "vid";
    n.type = ValueType::Int;
    return Int{ ast().append(std::move(n)) };
  }
};

struct Float {
  int id{ -1 };

  Float() = default;
  explicit Float(int node_id) : id(node_id) {}
  Float(const Float &) = default;
  Float &operator=(const Float &other);
  Float &operator+=(Float other);
  Float &operator-=(Float other);
  Float &operator*=(Float other);
  Float &operator/=(Float other);

  static Float constant(double value)
  {
    ExprNode n = ExprNode::make(OpKind::ConstFloat);
    n.const_f = value;
    n.type = ValueType::Float;
    return Float{ ast().append(std::move(n)) };
  }
};

struct Float2 {
  int id{ -1 };
  Float2() = default;
  explicit Float2(int node_id) : id(node_id) {}
  Float2(const Float2 &) = default;
  Float2 &operator=(const Float2 &other);
};

struct Float3 {
  int id{ -1 };
  Float3() = default;
  explicit Float3(int node_id) : id(node_id) {}
  Float3(const Float3 &) = default;
  Float3 &operator=(const Float3 &other);
};

struct Float4 {
  int id{ -1 };
  Float4() = default;
  explicit Float4(int node_id) : id(node_id) {}
  Float4(const Float4 &) = default;
  Float4 &operator=(const Float4 &other);
};

inline int bin(OpKind kind, int lhs, int rhs, ValueType type = ValueType::Float)
{
  ExprNode n = ExprNode::make(kind, lhs, rhs);
  n.type = type;
  return ast().append(std::move(n));
}

inline int un(OpKind kind, int lhs, ValueType type = ValueType::Float)
{
  ExprNode n = ExprNode::make(kind, lhs);
  n.type = type;
  return ast().append(std::move(n));
}

inline Int operator+(Int a, Int b) { return Int{ bin(OpKind::Add, a.id, b.id, ValueType::Int) }; }
inline Int operator-(Int a, Int b) { return Int{ bin(OpKind::Sub, a.id, b.id, ValueType::Int) }; }
inline Int operator*(Int a, Int b) { return Int{ bin(OpKind::Mul, a.id, b.id, ValueType::Int) }; }
inline Int operator/(Int a, Int b) { return Int{ bin(OpKind::Div, a.id, b.id, ValueType::Int) }; }
inline Int operator-(Int a) { return Int{ un(OpKind::Neg, a.id, ValueType::Int) }; }

inline Float operator+(Float a, Float b) { return Float{ bin(OpKind::Add, a.id, b.id) }; }
inline Float operator-(Float a, Float b) { return Float{ bin(OpKind::Sub, a.id, b.id) }; }
inline Float operator*(Float a, Float b) { return Float{ bin(OpKind::Mul, a.id, b.id) }; }
inline Float operator/(Float a, Float b) { return Float{ bin(OpKind::Div, a.id, b.id) }; }
inline Float operator-(Float a) { return Float{ un(OpKind::Neg, a.id) }; }

inline Float operator+(Float a, double b) { return a + Float::constant(b); }
inline Float operator-(Float a, double b) { return a - Float::constant(b); }
inline Float operator*(Float a, double b) { return a * Float::constant(b); }
inline Float operator/(Float a, double b) { return a / Float::constant(b); }
inline Float operator+(double a, Float b) { return Float::constant(a) + b; }
inline Float operator-(double a, Float b) { return Float::constant(a) - b; }
inline Float operator*(double a, Float b) { return Float::constant(a) * b; }
inline Float operator/(double a, Float b) { return Float::constant(a) / b; }

inline Bool operator<(Int a, Int b) { return Bool{ bin(OpKind::Less, a.id, b.id, ValueType::Bool) }; }
inline Bool operator<=(Int a, Int b) { return Bool{ bin(OpKind::LessEqual, a.id, b.id, ValueType::Bool) }; }
inline Bool operator>(Int a, Int b) { return Bool{ bin(OpKind::Greater, a.id, b.id, ValueType::Bool) }; }
inline Bool operator>=(Int a, Int b) { return Bool{ bin(OpKind::GreaterEqual, a.id, b.id, ValueType::Bool) }; }
inline Bool operator==(Int a, Int b) { return Bool{ bin(OpKind::Equal, a.id, b.id, ValueType::Bool) }; }
inline Bool operator!=(Int a, Int b) { return Bool{ bin(OpKind::NotEqual, a.id, b.id, ValueType::Bool) }; }

inline Bool operator<(Float a, Float b) { return Bool{ bin(OpKind::Less, a.id, b.id, ValueType::Bool) }; }
inline Bool operator<=(Float a, Float b) { return Bool{ bin(OpKind::LessEqual, a.id, b.id, ValueType::Bool) }; }
inline Bool operator>(Float a, Float b) { return Bool{ bin(OpKind::Greater, a.id, b.id, ValueType::Bool) }; }
inline Bool operator>=(Float a, Float b) { return Bool{ bin(OpKind::GreaterEqual, a.id, b.id, ValueType::Bool) }; }
inline Bool operator==(Float a, Float b) { return Bool{ bin(OpKind::Equal, a.id, b.id, ValueType::Bool) }; }
inline Bool operator!=(Float a, Float b) { return Bool{ bin(OpKind::NotEqual, a.id, b.id, ValueType::Bool) }; }

inline Bool operator&&(Bool a, Bool b) { return Bool{ bin(OpKind::LogicalAnd, a.id, b.id, ValueType::Bool) }; }
inline Bool operator||(Bool a, Bool b) { return Bool{ bin(OpKind::LogicalOr, a.id, b.id, ValueType::Bool) }; }
inline Bool operator!(Bool a) { return Bool{ un(OpKind::LogicalNot, a.id, ValueType::Bool) }; }

inline Float sin(Float x) { return Float{ un(OpKind::Sin, x.id) }; }
inline Float cos(Float x) { return Float{ un(OpKind::Cos, x.id) }; }
inline Float sqrt(Float x) { return Float{ un(OpKind::Sqrt, x.id) }; }
inline Float abs(Float x) { return Float{ un(OpKind::Abs, x.id) }; }
inline Float floor(Float x) { return Float{ un(OpKind::Floor, x.id) }; }
inline Float ceil(Float x) { return Float{ un(OpKind::Ceil, x.id) }; }
inline Float min(Float a, Float b) { return Float{ bin(OpKind::Min, a.id, b.id) }; }
inline Float max(Float a, Float b) { return Float{ bin(OpKind::Max, a.id, b.id) }; }

inline Float select(Bool cond, Float t, Float f)
{
  ExprNode n = ExprNode::make(OpKind::Select, cond.id, t.id, f.id);
  n.type = ValueType::Float;
  return Float{ ast().append(std::move(n)) };
}
inline Int select(Bool cond, Int t, Int f)
{
  ExprNode n = ExprNode::make(OpKind::Select, cond.id, t.id, f.id);
  n.type = ValueType::Int;
  return Int{ ast().append(std::move(n)) };
}
inline Float2 select(Bool cond, Float2 t, Float2 f)
{
  ExprNode n = ExprNode::make(OpKind::Select, cond.id, t.id, f.id);
  n.type = ValueType::Vec2;
  return Float2{ ast().append(std::move(n)) };
}
inline Float3 select(Bool cond, Float3 t, Float3 f)
{
  ExprNode n = ExprNode::make(OpKind::Select, cond.id, t.id, f.id);
  n.type = ValueType::Vec3;
  return Float3{ ast().append(std::move(n)) };
}
inline Float4 select(Bool cond, Float4 t, Float4 f)
{
  ExprNode n = ExprNode::make(OpKind::Select, cond.id, t.id, f.id);
  n.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(n)) };
}

inline Float2 vec2(Float x, Float y)
{
  ExprNode n = ExprNode::make(OpKind::Vec2, x.id, y.id);
  n.type = ValueType::Vec2;
  return Float2{ ast().append(std::move(n)) };
}
inline Float2 vec2(double x, double y) { return vec2(Float::constant(x), Float::constant(y)); }

inline Float3 vec3(Float x, Float y, Float z)
{
  ExprNode n = ExprNode::make(OpKind::Vec3, x.id, y.id, z.id);
  n.type = ValueType::Vec3;
  return Float3{ ast().append(std::move(n)) };
}
inline Float3 vec3(double x, double y, double z)
{
  return vec3(Float::constant(x), Float::constant(y), Float::constant(z));
}

inline Float4 vec4(Float x, Float y, Float z, Float w)
{
  ExprNode n = ExprNode::make(OpKind::Vec4, x.id, y.id, z.id, w.id);
  n.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(n)) };
}
inline Float4 vec4(Float2 v, Float z, Float w)
{
  ExprNode n = ExprNode::make(OpKind::Vec4From2, v.id, z.id, w.id);
  n.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(n)) };
}
inline Float4 vec4(Float3 v, Float w)
{
  ExprNode n = ExprNode::make(OpKind::Vec4From2, v.id, w.id); // reuse: vec4(vec3, float)
  n.kind = OpKind::Vec4From2;
  n.name = "vec3";
  n.type = ValueType::Vec4;
  return Float4{ ast().append(std::move(n)) };
}
inline Float4 vec4(Float3 v, double w) { return vec4(v, Float::constant(w)); }

inline void emit_assign(int dst_var, int src)
{
  ast().append(ExprNode::make(OpKind::Assign, dst_var, src));
}

inline Int &Int::operator=(const Int &other)
{
  if (id < 0) { id = ast().make_temp("i", ValueType::Int); }
  emit_assign(id, other.id);
  return *this;
}
inline Int &Int::operator+=(Int other) { return (*this = *this + other); }
inline Int &Int::operator-=(Int other) { return (*this = *this - other); }
inline Int &Int::operator*=(Int other) { return (*this = *this * other); }
inline Int &Int::operator/=(Int other) { return (*this = *this / other); }

inline Float &Float::operator=(const Float &other)
{
  if (id < 0) { id = ast().make_temp("f", ValueType::Float); }
  emit_assign(id, other.id);
  return *this;
}
inline Float &Float::operator+=(Float other) { return (*this = *this + other); }
inline Float &Float::operator-=(Float other) { return (*this = *this - other); }
inline Float &Float::operator*=(Float other) { return (*this = *this * other); }
inline Float &Float::operator/=(Float other) { return (*this = *this / other); }

inline Float2 &Float2::operator=(const Float2 &other)
{
  if (id < 0) { id = ast().make_temp("v2", ValueType::Vec2); }
  emit_assign(id, other.id);
  return *this;
}
inline Float3 &Float3::operator=(const Float3 &other)
{
  if (id < 0) { id = ast().make_temp("v3", ValueType::Vec3); }
  emit_assign(id, other.id);
  return *this;
}
inline Float4 &Float4::operator=(const Float4 &other)
{
  if (id < 0) { id = ast().make_temp("v4", ValueType::Vec4); }
  emit_assign(id, other.id);
  return *this;
}

/// Vertex-stage outputs written during tracing.
struct VertexWriter {
  void position(Float2 xy)
  {
    Float4 clip = vec4(xy, Float::constant(0.0), Float::constant(1.0));
    ExprNode n = ExprNode::make(OpKind::OutputVarying, -1, clip.id);
    n.name = "gl_Position";
    n.type = ValueType::Vec4;
    ast().append(std::move(n));
  }
  void position(Float4 clip)
  {
    ExprNode n = ExprNode::make(OpKind::OutputVarying, -1, clip.id);
    n.name = "gl_Position";
    n.type = ValueType::Vec4;
    ast().append(std::move(n));
  }
  void color(Float3 rgb)
  {
    ExprNode n = ExprNode::make(OpKind::OutputVarying, -1, rgb.id);
    n.name = "vColor";
    n.type = ValueType::Vec3;
    n.const_i = 0; // location
    ast().append(std::move(n));
  }
};

/// Fragment-stage interpolated inputs.
struct FragmentReader {
  [[nodiscard]] Float3 color() const
  {
    ExprNode n = ExprNode::make(OpKind::InputVarying);
    n.name = "vColor";
    n.type = ValueType::Vec3;
    n.const_i = 0;
    return Float3{ ast().append(std::move(n)) };
  }
};

/// Fragment-stage color attachment outputs.
struct FragmentWriter {
  void color(Float4 rgba)
  {
    ExprNode n = ExprNode::make(OpKind::OutputVarying, -1, rgba.id);
    n.name = "fragColor";
    n.type = ValueType::Vec4;
    n.const_i = 0;
    ast().append(std::move(n));
  }
};

} // namespace vlk
