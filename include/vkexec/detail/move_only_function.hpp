#ifndef VKEXEC_DETAIL_MOVE_ONLY_FUNCTION_HPP
#define VKEXEC_DETAIL_MOVE_ONLY_FUNCTION_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace vkexec::detail {

template<typename Sig> class move_only_function;

template<typename R, typename... Args> class move_only_function<R(Args...)>
{
public:
  move_only_function() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  move_only_function(std::nullptr_t) noexcept {}

  template<typename F>
    requires std::invocable<F &, Args...> && (!std::same_as<std::remove_cvref_t<F>, move_only_function>)
             && (!std::same_as<std::remove_cvref_t<F>, std::nullptr_t>)
  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  move_only_function(F &&callable)
  { init(std::forward<F>(callable)); }

  move_only_function(move_only_function &&other) noexcept { move_from(std::move(other)); }

  // cppcheck-suppress operatorEqVarError
  auto operator=(move_only_function &&other) noexcept -> move_only_function &
  {
    if (this != &other) {
      clear();
      move_from(std::move(other));
    }
    return *this;
  }

  template<typename F>
    requires std::invocable<F &, Args...> && (!std::same_as<std::remove_cvref_t<F>, move_only_function>)
             && (!std::same_as<std::remove_cvref_t<F>, std::nullptr_t>)
  auto operator=(F &&callable) -> move_only_function &
  {
    clear();
    init(std::forward<F>(callable));
    return *this;
  }

  auto operator=(std::nullptr_t) noexcept -> move_only_function &
  {
    clear();
    return *this;
  }

  ~move_only_function() { clear(); }

  move_only_function(move_only_function const &) = delete;
  auto operator=(move_only_function const &) -> move_only_function & = delete;

  explicit operator bool() const noexcept { return invoker_ != nullptr; }

  auto operator()(Args... args) -> R
  {
    assert(invoker_ != nullptr);
    return invoker_(storage(), std::forward<Args>(args)...);
  }

private:
  using invoker_fn = R (*)(void *, Args...);
  using relocator_fn = void (*)(void *, void *);
  using destructor_fn = void (*)(void *);

  static constexpr std::size_t k_buffer_size = 3 * sizeof(void *);

  alignas(std::max_align_t) unsigned char buffer_[k_buffer_size]{};
  invoker_fn invoker_{ nullptr };
  relocator_fn relocator_{ nullptr };
  destructor_fn destructor_{ nullptr };

  [[nodiscard]] auto storage() noexcept -> void * { return static_cast<void *>(buffer_); }
  [[nodiscard]] auto storage() const noexcept -> void const * { return static_cast<void const *>(buffer_); }

  template<typename F> static auto invoke(void *slot, Args... args) -> R
  {
    if constexpr (std::is_void_v<R>) {
      std::invoke(*static_cast<F *>(slot), std::forward<Args>(args)...);
    } else {
      return std::invoke(*static_cast<F *>(slot), std::forward<Args>(args)...);
    }
  }

  template<typename F>
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static void relocate(void *dst, void *src)
  {
    auto *from = static_cast<F *>(src);
    new (dst) F(std::move(*from));
    from->~F();
  }

  template<typename F> static void destroy_object(void *slot) { static_cast<F *>(slot)->~F(); }

  template<typename F> void init(F &&callable)
  {
    using fn = std::remove_cvref_t<F>;
    if constexpr (sizeof(fn) <= k_buffer_size && alignof(fn) <= alignof(std::max_align_t)
                  && std::is_nothrow_move_constructible_v<fn>) {
      new (storage()) fn(std::forward<F>(callable));
      invoker_ = invoke<fn>;
      relocator_ = relocate<fn>;
      destructor_ = destroy_object<fn>;
    } else {
      using stored = std::unique_ptr<fn>;
      new (storage()) stored(std::make_unique<fn>(std::forward<F>(callable)));
      invoker_ = [](void *slot, Args... args) -> R {
        if constexpr (std::is_void_v<R>) {
          std::invoke(**static_cast<stored *>(slot), std::forward<Args>(args)...);
        } else {
          return std::invoke(**static_cast<stored *>(slot), std::forward<Args>(args)...);
        }
      };
      relocator_ = relocate<stored>;
      destructor_ = destroy_object<stored>;
    }
  }

  void clear() noexcept
  {
    if (invoker_ != nullptr) {
      destructor_(const_cast<void *>(storage()));
      invoker_ = nullptr;
      relocator_ = nullptr;
      destructor_ = nullptr;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  void move_from(move_only_function &&other) noexcept
  {
    if (other.invoker_ == nullptr) { return; }
    invoker_fn const new_invoker = other.invoker_;
    relocator_fn const new_relocator = other.relocator_;
    destructor_fn const new_destructor = other.destructor_;
    new_relocator(storage(), other.storage());
    invoker_ = new_invoker;
    relocator_ = new_relocator;
    destructor_ = new_destructor;
    other.invoker_ = nullptr;
    other.relocator_ = nullptr;
    other.destructor_ = nullptr;
  }
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_MOVE_ONLY_FUNCTION_HPP
