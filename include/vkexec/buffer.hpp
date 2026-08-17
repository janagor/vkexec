#ifndef VKEXEC_BUFFER_HPP
#define VKEXEC_BUFFER_HPP


#include <vkexec/context.hpp>
#include <vkexec_edsl/types.hpp>

#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

template<typename T> class buffer
{
public:
  buffer(context &ctx, std::size_t count, T fill = T{})
    : ctx_(&ctx), count_(count), name_("buf" + std::to_string(next_name_id()))
  {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count == 0) { throw std::invalid_argument("vkexec::buffer count must be > 0"); }

    auto bytes = static_cast<VkDeviceSize>(count * sizeof(T));

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VmaAllocationInfo ainfo{};
    if (vmaCreateBuffer(ctx_->allocator(), &bci, &aci, &buffer_, &allocation_, &ainfo) != VK_SUCCESS) {
      throw std::runtime_error("vmaCreateBuffer failed");
    }
    mapped_ = ainfo.pMappedData;
    if (mapped_ == nullptr) { throw std::runtime_error("vmaCreateBuffer did not map host-visible memory"); }

    auto *const elems = static_cast<T *>(mapped_);
    for (T &elem : std::span<T>{ elems, count_ }) { elem = fill; }
  }

  ~buffer()
  {
    if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
    }
  }

  buffer(buffer const &) = delete;
  auto operator=(buffer const &) -> buffer & = delete;

  buffer(buffer &&other) noexcept
    : ctx_(other.ctx_), buffer_(other.buffer_), allocation_(other.allocation_), mapped_(other.mapped_),
      count_(other.count_), name_(std::move(other.name_))
  {
    other.ctx_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
  }

  auto operator=(buffer &&other) noexcept -> buffer &
  {
    if (this == &other) { return *this; }
    if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
    }
    ctx_ = other.ctx_;
    buffer_ = other.buffer_;
    allocation_ = other.allocation_;
    mapped_ = other.mapped_;
    count_ = other.count_;
    name_ = std::move(other.name_);
    binding_ = -1;
    other.ctx_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
    other.count_ = 0;
    other.binding_ = -1;
    return *this;
  }

  [[nodiscard]] auto data() noexcept -> T * { return static_cast<T *>(mapped_); }
  [[nodiscard]] auto data() const noexcept -> T const * { return static_cast<T const *>(mapped_); }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return count_; }
  [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer { return buffer_; }
  [[nodiscard]] auto name() const noexcept -> std::string const & { return name_; }

  struct ref
  {
    buffer *owner;
    edsl::Int index;

    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions) -- eDSL implicit load
    operator edsl::Float() const
      requires(std::is_floating_point_v<T>)
    { return load_float(); }
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions) -- eDSL implicit load
    operator edsl::Int() const
      requires(std::is_integral_v<T>)
    { return load_int(); }

    auto operator=(edsl::Float value) -> ref &
      requires(std::is_floating_point_v<T>)
    {
      store(value.id);
      return *this;
    }
    auto operator=(edsl::Int value) -> ref &
      requires(std::is_integral_v<T>)
    {
      store(value.id);
      return *this;
    }

  private:
    [[nodiscard]] auto ensure_binding() const -> int
    {
      auto &ast_ctx = edsl::ast();
      if (owner->binding_ < 0) { owner->binding_ = ast_ctx.next_binding++; }
      for (auto &existing : ast_ctx.buffers) {
        // cppcheck-suppress useStlAlgorithm
        if (existing.binding == owner->binding_) {
          existing.vk_buffer = owner->buffer_;
          existing.byte_size = owner->count_ * sizeof(T);
          existing.elem_count = owner->count_;
          return owner->binding_;
        }
      }
      edsl::BufferBinding binding_info;
      binding_info.name = owner->name_;
      binding_info.elem_glsl_type = std::is_floating_point_v<T> ? "float" : "int";
      binding_info.binding = owner->binding_;
      binding_info.vk_buffer = owner->buffer_;
      binding_info.byte_size = owner->count_ * sizeof(T);
      binding_info.elem_count = owner->count_;
      ast_ctx.buffers.push_back(binding_info);
      if (ast_ctx.next_binding <= owner->binding_) { ast_ctx.next_binding = owner->binding_ + 1; }
      return owner->binding_;
    }

    // cppcheck-suppress unusedPrivateFunction
    [[nodiscard]] auto load_float() const -> edsl::Float
    {
      int const binding = ensure_binding();
      edsl::ExprNode node = edsl::ExprNode::make(edsl::OpKind::Load, index.id);
      node.binding = binding;
      int const load = edsl::ast().append(std::move(node));
      int const tmp = edsl::ast().make_temp("f");
      edsl::emit_assign(tmp, load);
      return edsl::Float{ tmp };
    }
    // cppcheck-suppress unusedPrivateFunction
    [[nodiscard]] auto load_int() const -> edsl::Int
    {
      int const binding = ensure_binding();
      edsl::ExprNode node = edsl::ExprNode::make(edsl::OpKind::Load, index.id);
      node.binding = binding;
      int const load = edsl::ast().append(std::move(node));
      int const tmp = edsl::ast().make_temp("i");
      edsl::emit_assign(tmp, load);
      return edsl::Int{ tmp };
    }
    void store(int value_id) const
    {
      int const binding = ensure_binding();
      edsl::ExprNode node = edsl::ExprNode::make(edsl::OpKind::Store, index.id, value_id);
      node.binding = binding;
      edsl::ast().append(std::move(node));
    }
  };

  auto operator[](edsl::Int idx) -> ref { return ref{ this, idx }; }

  void register_for_dispatch(edsl::ASTContext &ast_ctx) const
  {
    if (binding_ < 0) { return; }
    for (auto &entry : ast_ctx.buffers) {
      if (entry.binding == binding_) {
        entry.vk_buffer = buffer_;
        entry.byte_size = count_ * sizeof(T);
        entry.elem_count = count_;
      }
    }
  }

private:
  static auto next_name_id() -> int
  {
    static int name_id = 0;
    return name_id++;
  }

  context *ctx_{ nullptr };
  VkBuffer buffer_{ VK_NULL_HANDLE };
  VmaAllocation allocation_{ VK_NULL_HANDLE };
  void *mapped_{ nullptr };
  std::size_t count_{ 0 };
  std::string name_;
  mutable int binding_{ -1 };
};

}// namespace vkexec

#endif// VKEXEC_BUFFER_HPP
