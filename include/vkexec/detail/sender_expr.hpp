#ifndef VKEXEC_DETAIL_SENDER_EXPR_HPP
#define VKEXEC_DETAIL_SENDER_EXPR_HPP

#include <stdexec/execution.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace vkexec::detail {

namespace ex = stdexec;

struct empty_data
{
};

template<class Tag, class Data, class Child> struct sender_expr
{
  using sender_concept = ex::sender_t;

  [[no_unique_address]] Tag tag;
  [[no_unique_address]] Data data;
  [[no_unique_address]] Child child;

  // Semantic expressions are transparent to scheduler and domain queries
  // until vkexec::domain lowers them into an execution sender.
  [[nodiscard]] auto get_env() const noexcept(noexcept(ex::get_env(child))) -> decltype(ex::get_env(child))
  { return ex::get_env(child); }
};

template<class Tag, class Data, class Child>
[[nodiscard]] auto make_sender_expr(Tag tag, Data &&data, Child &&child) noexcept(
  std::is_nothrow_move_constructible_v<Tag> && std::is_nothrow_constructible_v<std::decay_t<Data>, Data>
  && std::is_nothrow_constructible_v<std::decay_t<Child>, Child>)
  -> sender_expr<Tag, std::decay_t<Data>, std::decay_t<Child>>
{ return { .tag = std::move(tag), .data = std::forward<Data>(data), .child = std::forward<Child>(child) }; }

template<class T> struct is_sender_expr : std::false_type
{
};

template<class Tag, class Data, class Child> struct is_sender_expr<sender_expr<Tag, Data, Child>> : std::true_type
{
};

template<class T>
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr bool is_sender_expr_v = is_sender_expr<std::remove_cvref_t<T>>::value;

template<class T> struct sender_expr_traits;

template<class Tag, class Data, class Child> struct sender_expr_traits<sender_expr<Tag, Data, Child>>
{
  using tag_type = Tag;
  using data_type = Data;
  using child_type = Child;
};

template<class T> using expression_tag_t = sender_expr_traits<std::remove_cvref_t<T>>::tag_type;

template<class Tag, class Data> struct expr_closure : ex::sender_adaptor_closure<expr_closure<Tag, Data>>
{
  [[no_unique_address]] Tag tag;
  [[no_unique_address]] Data data;

  template<ex::sender Sender>
  [[nodiscard]] auto operator()(Sender &&sender) && noexcept(
    noexcept(make_sender_expr(std::move(tag), std::move(data), std::forward<Sender>(sender))))
    -> sender_expr<Tag, Data, std::decay_t<Sender>>
  { return make_sender_expr(std::move(tag), std::move(data), std::forward<Sender>(sender)); }

  template<ex::sender Sender>
    requires std::copy_constructible<Tag> && std::copy_constructible<Data>
  [[nodiscard]] auto operator()(Sender &&sender) const & noexcept(
    noexcept(make_sender_expr(tag, data, std::forward<Sender>(sender)))) -> sender_expr<Tag, Data, std::decay_t<Sender>>
  { return make_sender_expr(tag, data, std::forward<Sender>(sender)); }
};

template<class Tag, class Data>
[[nodiscard]] auto make_expr_closure(Tag tag, Data &&data) noexcept(
  std::is_nothrow_move_constructible_v<Tag> && std::is_nothrow_constructible_v<std::decay_t<Data>, Data>)
  -> expr_closure<Tag, std::decay_t<Data>>
{ return { {}, std::move(tag), std::forward<Data>(data) }; }

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SENDER_EXPR_HPP
