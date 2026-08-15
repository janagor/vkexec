#pragma once

#include <vkexec/detail/ast.hpp>

#include <cstdint>

namespace vlk {

struct Bool;
struct Int;
struct Float;

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
    return Int{ ast().append(std::move(n)) };
  }

  static Int param_index()
  {
    ExprNode n = ExprNode::make(OpKind::ParamIndex);
    n.name = "idx";
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
    return Float{ ast().append(std::move(n)) };
  }
};

inline int bin(OpKind kind, int lhs, int rhs) { return ast().append(ExprNode::make(kind, lhs, rhs)); }

inline int un(OpKind kind, int lhs) { return ast().append(ExprNode::make(kind, lhs)); }

inline Int operator+(Int a, Int b) { return Int{ bin(OpKind::Add, a.id, b.id) }; }
inline Int operator-(Int a, Int b) { return Int{ bin(OpKind::Sub, a.id, b.id) }; }
inline Int operator*(Int a, Int b) { return Int{ bin(OpKind::Mul, a.id, b.id) }; }
inline Int operator/(Int a, Int b) { return Int{ bin(OpKind::Div, a.id, b.id) }; }
inline Int operator-(Int a) { return Int{ un(OpKind::Neg, a.id) }; }

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

inline Bool operator<(Int a, Int b) { return Bool{ bin(OpKind::Less, a.id, b.id) }; }
inline Bool operator<=(Int a, Int b) { return Bool{ bin(OpKind::LessEqual, a.id, b.id) }; }
inline Bool operator>(Int a, Int b) { return Bool{ bin(OpKind::Greater, a.id, b.id) }; }
inline Bool operator>=(Int a, Int b) { return Bool{ bin(OpKind::GreaterEqual, a.id, b.id) }; }
inline Bool operator==(Int a, Int b) { return Bool{ bin(OpKind::Equal, a.id, b.id) }; }
inline Bool operator!=(Int a, Int b) { return Bool{ bin(OpKind::NotEqual, a.id, b.id) }; }

inline Bool operator<(Float a, Float b) { return Bool{ bin(OpKind::Less, a.id, b.id) }; }
inline Bool operator<=(Float a, Float b) { return Bool{ bin(OpKind::LessEqual, a.id, b.id) }; }
inline Bool operator>(Float a, Float b) { return Bool{ bin(OpKind::Greater, a.id, b.id) }; }
inline Bool operator>=(Float a, Float b) { return Bool{ bin(OpKind::GreaterEqual, a.id, b.id) }; }
inline Bool operator==(Float a, Float b) { return Bool{ bin(OpKind::Equal, a.id, b.id) }; }
inline Bool operator!=(Float a, Float b) { return Bool{ bin(OpKind::NotEqual, a.id, b.id) }; }

inline Bool operator&&(Bool a, Bool b) { return Bool{ bin(OpKind::LogicalAnd, a.id, b.id) }; }
inline Bool operator||(Bool a, Bool b) { return Bool{ bin(OpKind::LogicalOr, a.id, b.id) }; }
inline Bool operator!(Bool a) { return Bool{ un(OpKind::LogicalNot, a.id) }; }

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
  return Float{ ast().append(ExprNode::make(OpKind::Select, cond.id, t.id, f.id)) };
}

inline void emit_assign(int dst_var, int src)
{
  ast().append(ExprNode::make(OpKind::Assign, dst_var, src));
}

inline Int &Int::operator=(const Int &other)
{
  if (id < 0) { id = ast().make_temp("i"); }
  emit_assign(id, other.id);
  return *this;
}
inline Int &Int::operator+=(Int other) { return (*this = *this + other); }
inline Int &Int::operator-=(Int other) { return (*this = *this - other); }
inline Int &Int::operator*=(Int other) { return (*this = *this * other); }
inline Int &Int::operator/=(Int other) { return (*this = *this / other); }

inline Float &Float::operator=(const Float &other)
{
  if (id < 0) { id = ast().make_temp("f"); }
  emit_assign(id, other.id);
  return *this;
}
inline Float &Float::operator+=(Float other) { return (*this = *this + other); }
inline Float &Float::operator-=(Float other) { return (*this = *this - other); }
inline Float &Float::operator*=(Float other) { return (*this = *this * other); }
inline Float &Float::operator/=(Float other) { return (*this = *this / other); }

} // namespace vlk
