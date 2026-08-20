#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/ast.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/trace_access.hpp>
#include <vkexec_edsl/types.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace edsl = vkexec::edsl;

namespace {

constexpr std::int64_t k_int_one = 1;
constexpr std::int64_t k_int_two = 2;
constexpr double k_float_one = 1.0;
constexpr double k_float_two = 2.0;
constexpr double k_float_three = 3.0;
constexpr double k_float_four = 4.0;

[[nodiscard]] auto node_at(edsl::trace_scope const &scope, int node_id) -> edsl::ExprNode const &
{ return edsl::detail::trace_ast_access::get(scope).nodes.at(static_cast<std::size_t>(node_id)); }

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto make_vec4(double coord_x, double coord_y, double coord_z, double coord_w) -> edsl::Float4
{
  return edsl::vec4(edsl::Float::constant(coord_x),
    edsl::Float::constant(coord_y),
    edsl::Float::constant(coord_z),
    edsl::Float::constant(coord_w));
}
// NOLINTEND(bugprone-easily-swappable-parameters)

template<typename T> auto assign_copy(T &lhs, T const &rhs) -> void { lhs = rhs; }

}// namespace

TEST_CASE("eDSL constants record typed AST nodes", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;

  edsl::Bool const true_value = edsl::Bool::constant(true);
  edsl::Bool const false_value = edsl::Bool::constant(false);
  edsl::Int const int_value = edsl::Int::constant(k_int_two);
  edsl::Float const float_value = edsl::Float::constant(k_float_one);
  edsl::Int const idx = edsl::Int::param_index();
  edsl::Int const vid = edsl::Int::vertex_index();

  edsl::ExprNode const &true_node = node_at(scope, true_value.id);
  REQUIRE(true_node.kind == edsl::OpKind::ConstBool);
  REQUIRE(true_node.type == edsl::ValueType::Bool);
  REQUIRE(true_node.const_i == 1);

  edsl::ExprNode const &false_node = node_at(scope, false_value.id);
  REQUIRE(false_node.kind == edsl::OpKind::ConstBool);
  REQUIRE(false_node.const_i == 0);

  edsl::ExprNode const &int_node = node_at(scope, int_value.id);
  REQUIRE(int_node.kind == edsl::OpKind::ConstInt);
  REQUIRE(int_node.type == edsl::ValueType::Int);
  REQUIRE(int_node.const_i == k_int_two);

  edsl::ExprNode const &float_node = node_at(scope, float_value.id);
  REQUIRE(float_node.kind == edsl::OpKind::ConstFloat);
  REQUIRE(float_node.type == edsl::ValueType::Float);
  REQUIRE(float_node.const_f == k_float_one);

  edsl::ExprNode const &idx_node = node_at(scope, idx.id);
  REQUIRE(idx_node.kind == edsl::OpKind::ParamIndex);
  REQUIRE(idx_node.name == "idx");
  REQUIRE(idx_node.type == edsl::ValueType::Int);

  edsl::ExprNode const &vid_node = node_at(scope, vid.id);
  REQUIRE(vid_node.kind == edsl::OpKind::VertexIndex);
  REQUIRE(vid_node.name == "vid");
  REQUIRE(vid_node.type == edsl::ValueType::Int);
}

TEST_CASE("eDSL integer arithmetic records binary and unary ops", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  edsl::Int const lhs = edsl::Int::constant(k_int_one);
  edsl::Int const rhs = edsl::Int::constant(k_int_two);

  edsl::Int const sum = lhs + rhs;
  edsl::Int const diff = lhs - rhs;
  edsl::Int const prod = lhs * rhs;
  edsl::Int const quot = lhs / rhs;
  edsl::Int const neg = -lhs;

  REQUIRE(node_at(scope, sum.id).kind == edsl::OpKind::Add);
  REQUIRE(node_at(scope, sum.id).type == edsl::ValueType::Int);
  REQUIRE(node_at(scope, diff.id).kind == edsl::OpKind::Sub);
  REQUIRE(node_at(scope, prod.id).kind == edsl::OpKind::Mul);
  REQUIRE(node_at(scope, quot.id).kind == edsl::OpKind::Div);
  REQUIRE(node_at(scope, neg.id).kind == edsl::OpKind::Neg);
  REQUIRE(node_at(scope, neg.id).type == edsl::ValueType::Int);
  REQUIRE(node_at(scope, neg.id).lhs == lhs.id);
}

TEST_CASE("eDSL float arithmetic records mixed double operators", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  edsl::Float const lhs = edsl::Float::constant(k_float_one);
  edsl::Float const rhs = edsl::Float::constant(k_float_two);

  edsl::Float const sum = lhs + rhs;
  edsl::Float const diff = lhs - rhs;
  edsl::Float const prod = lhs * rhs;
  edsl::Float const quot = lhs / rhs;
  edsl::Float const neg = -lhs;
  edsl::Float const lhs_plus = lhs + k_float_two;
  edsl::Float const lhs_minus = lhs - k_float_two;
  edsl::Float const lhs_times = lhs * k_float_two;
  edsl::Float const lhs_div = lhs / k_float_two;
  edsl::Float const rhs_plus = k_float_two + lhs;
  edsl::Float const rhs_minus = k_float_two - lhs;
  edsl::Float const rhs_times = k_float_two * lhs;
  edsl::Float const rhs_div = k_float_two / lhs;

  REQUIRE(node_at(scope, sum.id).kind == edsl::OpKind::Add);
  REQUIRE(node_at(scope, sum.id).type == edsl::ValueType::Float);
  REQUIRE(node_at(scope, diff.id).kind == edsl::OpKind::Sub);
  REQUIRE(node_at(scope, prod.id).kind == edsl::OpKind::Mul);
  REQUIRE(node_at(scope, quot.id).kind == edsl::OpKind::Div);
  REQUIRE(node_at(scope, neg.id).kind == edsl::OpKind::Neg);
  REQUIRE(node_at(scope, lhs_plus.id).kind == edsl::OpKind::Add);
  REQUIRE(node_at(scope, lhs_minus.id).kind == edsl::OpKind::Sub);
  REQUIRE(node_at(scope, lhs_times.id).kind == edsl::OpKind::Mul);
  REQUIRE(node_at(scope, lhs_div.id).kind == edsl::OpKind::Div);
  REQUIRE(node_at(scope, rhs_plus.id).kind == edsl::OpKind::Add);
  REQUIRE(node_at(scope, rhs_minus.id).kind == edsl::OpKind::Sub);
  REQUIRE(node_at(scope, rhs_times.id).kind == edsl::OpKind::Mul);
  REQUIRE(node_at(scope, rhs_div.id).kind == edsl::OpKind::Div);
}

TEST_CASE("eDSL comparisons and logic record boolean ops", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  edsl::Int const lhs_int = edsl::Int::constant(k_int_one);
  edsl::Int const rhs_int = edsl::Int::constant(k_int_two);
  edsl::Float const lhs_float = edsl::Float::constant(k_float_one);
  edsl::Float const rhs_float = edsl::Float::constant(k_float_two);

  edsl::Bool const int_lt = lhs_int < rhs_int;
  edsl::Bool const int_le = lhs_int <= rhs_int;
  edsl::Bool const int_gt = lhs_int > rhs_int;
  edsl::Bool const int_ge = lhs_int >= rhs_int;
  edsl::Bool const int_eq = lhs_int == rhs_int;
  edsl::Bool const int_ne = lhs_int != rhs_int;
  edsl::Bool const float_lt = lhs_float < rhs_float;
  edsl::Bool const float_le = lhs_float <= rhs_float;
  edsl::Bool const float_gt = lhs_float > rhs_float;
  edsl::Bool const float_ge = lhs_float >= rhs_float;
  edsl::Bool const float_eq = lhs_float == rhs_float;
  edsl::Bool const float_ne = lhs_float != rhs_float;
  edsl::Bool const both = int_lt && float_lt;
  edsl::Bool const either = int_gt || float_gt;
  edsl::Bool const not_lt = !int_lt;

  REQUIRE(node_at(scope, int_lt.id).kind == edsl::OpKind::Less);
  REQUIRE(node_at(scope, int_lt.id).type == edsl::ValueType::Bool);
  REQUIRE(node_at(scope, int_le.id).kind == edsl::OpKind::LessEqual);
  REQUIRE(node_at(scope, int_gt.id).kind == edsl::OpKind::Greater);
  REQUIRE(node_at(scope, int_ge.id).kind == edsl::OpKind::GreaterEqual);
  REQUIRE(node_at(scope, int_eq.id).kind == edsl::OpKind::Equal);
  REQUIRE(node_at(scope, int_ne.id).kind == edsl::OpKind::NotEqual);
  REQUIRE(node_at(scope, float_lt.id).kind == edsl::OpKind::Less);
  REQUIRE(node_at(scope, float_le.id).kind == edsl::OpKind::LessEqual);
  REQUIRE(node_at(scope, float_gt.id).kind == edsl::OpKind::Greater);
  REQUIRE(node_at(scope, float_ge.id).kind == edsl::OpKind::GreaterEqual);
  REQUIRE(node_at(scope, float_eq.id).kind == edsl::OpKind::Equal);
  REQUIRE(node_at(scope, float_ne.id).kind == edsl::OpKind::NotEqual);
  REQUIRE(node_at(scope, both.id).kind == edsl::OpKind::LogicalAnd);
  REQUIRE(node_at(scope, either.id).kind == edsl::OpKind::LogicalOr);
  REQUIRE(node_at(scope, not_lt.id).kind == edsl::OpKind::LogicalNot);
}

TEST_CASE("eDSL math helpers record unary and binary functions", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  edsl::Float const value = edsl::Float::constant(k_float_one);
  edsl::Float const other = edsl::Float::constant(k_float_two);

  REQUIRE(node_at(scope, edsl::sin(value).id).kind == edsl::OpKind::Sin);
  REQUIRE(node_at(scope, edsl::cos(value).id).kind == edsl::OpKind::Cos);
  REQUIRE(node_at(scope, edsl::sqrt(value).id).kind == edsl::OpKind::Sqrt);
  REQUIRE(node_at(scope, edsl::abs(value).id).kind == edsl::OpKind::Abs);
  REQUIRE(node_at(scope, edsl::floor(value).id).kind == edsl::OpKind::Floor);
  REQUIRE(node_at(scope, edsl::ceil(value).id).kind == edsl::OpKind::Ceil);
  REQUIRE(node_at(scope, edsl::min(value, other).id).kind == edsl::OpKind::Min);
  REQUIRE(node_at(scope, edsl::max(value, other).id).kind == edsl::OpKind::Max);
}

TEST_CASE("eDSL select records typed ternary nodes", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  edsl::Bool const cond = edsl::Bool::constant(true);
  edsl::Float const f_true = edsl::Float::constant(k_float_one);
  edsl::Float const f_false = edsl::Float::constant(k_float_two);
  edsl::Int const i_true = edsl::Int::constant(k_int_one);
  edsl::Int const i_false = edsl::Int::constant(k_int_two);
  edsl::Float2 const v2_true = edsl::vec2(k_float_one, k_float_two);
  edsl::Float2 const v2_false = edsl::vec2(k_float_two, k_float_one);
  edsl::Float3 const vec3_true = edsl::vec3(k_float_one, k_float_two, k_float_three);
  edsl::Float3 const vec3_false = edsl::vec3(k_float_two, k_float_one, k_float_three);
  edsl::Float4 const vec4_true = make_vec4(k_float_one, k_float_two, k_float_three, k_float_four);
  edsl::Float4 const vec4_false = make_vec4(k_float_two, k_float_one, k_float_three, k_float_four);

  edsl::Float const selected_f = edsl::select(cond, f_true, f_false);
  edsl::Int const selected_i = edsl::select(cond, i_true, i_false);
  edsl::Float2 const selected_v2 = edsl::select(cond, v2_true, v2_false);
  edsl::Float3 const selected_v3 = edsl::select(cond, vec3_true, vec3_false);
  edsl::Float4 const selected_v4 = edsl::select(cond, vec4_true, vec4_false);

  REQUIRE(node_at(scope, selected_f.id).kind == edsl::OpKind::Select);
  REQUIRE(node_at(scope, selected_f.id).type == edsl::ValueType::Float);
  REQUIRE(node_at(scope, selected_i.id).type == edsl::ValueType::Int);
  REQUIRE(node_at(scope, selected_v2.id).type == edsl::ValueType::Vec2);
  REQUIRE(node_at(scope, selected_v3.id).type == edsl::ValueType::Vec3);
  REQUIRE(node_at(scope, selected_v4.id).type == edsl::ValueType::Vec4);
}

TEST_CASE("eDSL vector constructors record Vec nodes", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  edsl::Float const coord_x = edsl::Float::constant(k_float_one);
  edsl::Float const coord_y = edsl::Float::constant(k_float_two);
  edsl::Float const coord_z = edsl::Float::constant(k_float_three);
  edsl::Float const coord_w = edsl::Float::constant(k_float_four);

  edsl::Float2 const from_floats = edsl::vec2(coord_x, coord_y);
  edsl::Float2 const from_doubles = edsl::vec2(k_float_one, k_float_two);
  edsl::Float3 const vec3_value = edsl::vec3(coord_x, coord_y, coord_z);
  edsl::Float3 const vec3_from_doubles = edsl::vec3(k_float_one, k_float_two, k_float_three);
  edsl::Float4 const vec4_value = edsl::vec4(coord_x, coord_y, coord_z, coord_w);
  edsl::Float4 const vec4_from2 = edsl::vec4(from_floats, coord_z, coord_w);
  edsl::Float4 const vec4_from3 = edsl::vec4(vec3_value, coord_w);
  edsl::Float4 const vec4_from3_double = edsl::vec4(vec3_value, k_float_four);

  REQUIRE(node_at(scope, from_floats.id).kind == edsl::OpKind::Vec2);
  REQUIRE(node_at(scope, from_floats.id).type == edsl::ValueType::Vec2);
  REQUIRE(node_at(scope, from_doubles.id).kind == edsl::OpKind::Vec2);
  REQUIRE(node_at(scope, vec3_value.id).kind == edsl::OpKind::Vec3);
  REQUIRE(node_at(scope, vec3_from_doubles.id).kind == edsl::OpKind::Vec3);
  REQUIRE(node_at(scope, vec4_value.id).kind == edsl::OpKind::Vec4);
  REQUIRE(node_at(scope, vec4_value.id).type == edsl::ValueType::Vec4);
  REQUIRE(node_at(scope, vec4_from2.id).kind == edsl::OpKind::Vec4From2);
  REQUIRE(node_at(scope, vec4_from3.id).kind == edsl::OpKind::Vec4From2);
  REQUIRE(node_at(scope, vec4_from3.id).name == "vec3");
  REQUIRE(node_at(scope, vec4_from3_double.id).kind == edsl::OpKind::Vec4From2);
}

TEST_CASE("eDSL integer assignment and compound ops", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;

  edsl::Int dest_int;
  dest_int = edsl::Int::constant(k_int_one);
  assign_copy(dest_int, dest_int);
  dest_int += edsl::Int::constant(k_int_one);
  dest_int -= edsl::Int::constant(k_int_one);
  dest_int *= edsl::Int::constant(k_int_two);
  dest_int /= edsl::Int::constant(k_int_two);
  REQUIRE(dest_int.id >= 0);

  edsl::Int move_src = edsl::Int::constant(k_int_one);
  edsl::Int moved_int;
  moved_int = std::move(move_src);
  REQUIRE(moved_int.id >= 0);
}

TEST_CASE("eDSL float assignment and compound ops", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;

  edsl::Float dest_float;
  dest_float = edsl::Float::constant(k_float_one);
  assign_copy(dest_float, dest_float);
  dest_float += edsl::Float::constant(k_float_one);
  dest_float -= edsl::Float::constant(k_float_one);
  dest_float *= edsl::Float::constant(k_float_two);
  dest_float /= edsl::Float::constant(k_float_two);
  REQUIRE(dest_float.id >= 0);

  edsl::Float move_src = edsl::Float::constant(k_float_one);
  edsl::Float moved_float;
  moved_float = std::move(move_src);
  REQUIRE(moved_float.id >= 0);
}

TEST_CASE("eDSL vector assignment allocates temps", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;

  edsl::Float2 dest_v2;
  dest_v2 = edsl::vec2(k_float_one, k_float_two);
  assign_copy(dest_v2, dest_v2);
  REQUIRE(dest_v2.id >= 0);

  edsl::Float3 dest_v3;
  dest_v3 = edsl::vec3(k_float_one, k_float_two, k_float_three);
  assign_copy(dest_v3, dest_v3);
  REQUIRE(dest_v3.id >= 0);

  edsl::Float4 dest_v4;
  dest_v4 = make_vec4(k_float_one, k_float_two, k_float_three, k_float_four);
  assign_copy(dest_v4, dest_v4);
  REQUIRE(dest_v4.id >= 0);

  edsl::Float2 move_src_v2 = edsl::vec2(k_float_one, k_float_two);
  edsl::Float2 moved_v2;
  moved_v2 = std::move(move_src_v2);
  REQUIRE(moved_v2.id >= 0);

  edsl::Float3 move_src_v3 = edsl::vec3(k_float_one, k_float_two, k_float_three);
  edsl::Float3 moved_v3;
  moved_v3 = std::move(move_src_v3);
  REQUIRE(moved_v3.id >= 0);

  edsl::Float4 move_src_v4 = make_vec4(k_float_one, k_float_two, k_float_three, k_float_four);
  edsl::Float4 moved_v4;
  moved_v4 = std::move(move_src_v4);
  REQUIRE(moved_v4.id >= 0);
}

TEST_CASE("eDSL vertex I/O records input and output varyings", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;

  edsl::Float3 const in_pos = edsl::VertexIn::position();
  REQUIRE(node_at(scope, in_pos.id).name == "inPosition");
  REQUIRE(node_at(scope, edsl::VertexIn::color().id).name == "inColor");

  edsl::VertexWriter::position(edsl::vec2(k_float_one, k_float_two));
  REQUIRE(edsl::detail::trace_ast_access::get(scope).nodes.back().name == "gl_Position");
  edsl::VertexWriter::point_size(edsl::Float::constant(k_float_one));
  REQUIRE(edsl::detail::trace_ast_access::get(scope).nodes.back().name == "gl_PointSize");
  edsl::VertexWriter::color(edsl::vec3(k_float_one, k_float_two, k_float_three));
  REQUIRE(edsl::detail::trace_ast_access::get(scope).nodes.back().name == "vColor");
  edsl::VertexWriter::color(make_vec4(k_float_one, k_float_two, k_float_three, k_float_four));
  REQUIRE(edsl::detail::trace_ast_access::get(scope).nodes.back().name == "vColor4");
}

TEST_CASE("eDSL fragment I/O records color varyings", "[vkexec][edsl][types]")
{
  edsl::trace_scope const scope;
  REQUIRE(node_at(scope, edsl::FragmentReader::color().id).name == "vColor");
  edsl::Float4 const frag_color4 = edsl::FragmentReader::color4();
  REQUIRE(node_at(scope, frag_color4.id).name == "vColor4");
  edsl::FragmentWriter::color(frag_color4);
  REQUIRE(edsl::detail::trace_ast_access::get(scope).nodes.back().name == "fragColor");
}
