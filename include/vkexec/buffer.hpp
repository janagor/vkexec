#pragma once

#include <vkexec/context.hpp>
#include <vkexec/detail/types.hpp>

#include <vulkan/vulkan.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace vkexec {

template<typename T>
class buffer {
public:
  buffer(context &ctx, std::size_t count, T fill = T{})
    : ctx_(&ctx), count_(count), name_("buf" + std::to_string(next_name_id()))
  {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count == 0) { throw std::invalid_argument("vkexec::buffer count must be > 0"); }

    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count * sizeof(T));

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_->device(), &bci, nullptr, &buffer_) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateBuffer failed");
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_->device(), buffer_, &req);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(ctx_->device(), &mai, nullptr, &memory_) != VK_SUCCESS) {
      throw std::runtime_error("vkAllocateMemory failed");
    }
    vkBindBufferMemory(ctx_->device(), buffer_, memory_, 0);
    if (vkMapMemory(ctx_->device(), memory_, 0, bytes, 0, &mapped_) != VK_SUCCESS) {
      throw std::runtime_error("vkMapMemory failed");
    }

    auto *ptr = static_cast<T *>(mapped_);
    for (std::size_t i = 0; i < count_; ++i) { ptr[i] = fill; }
  }

  ~buffer()
  {
    if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
      if (mapped_) { vkUnmapMemory(ctx_->device(), memory_); }
      if (buffer_ != VK_NULL_HANDLE) { vkDestroyBuffer(ctx_->device(), buffer_, nullptr); }
      if (memory_ != VK_NULL_HANDLE) { vkFreeMemory(ctx_->device(), memory_, nullptr); }
    }
  }

  buffer(const buffer &) = delete;
  buffer &operator=(const buffer &) = delete;

  buffer(buffer &&other) noexcept
    : ctx_(other.ctx_), buffer_(other.buffer_), memory_(other.memory_), mapped_(other.mapped_), count_(other.count_),
      name_(std::move(other.name_)), binding_(-1)
  {
    other.ctx_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
  }

  [[nodiscard]] T *data() noexcept { return static_cast<T *>(mapped_); }
  [[nodiscard]] const T *data() const noexcept { return static_cast<const T *>(mapped_); }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] VkBuffer vk_buffer() const noexcept { return buffer_; }
  [[nodiscard]] const std::string &name() const noexcept { return name_; }

  struct Ref {
    buffer *owner;
    vlk::Int index;

    operator vlk::Float() const
      requires(std::is_floating_point_v<T>)
    {
      return load_float();
    }
    operator vlk::Int() const
      requires(std::is_integral_v<T>)
    {
      return load_int();
    }

    Ref &operator=(vlk::Float value)
      requires(std::is_floating_point_v<T>)
    {
      store(value.id);
      return *this;
    }
    Ref &operator=(vlk::Int value)
      requires(std::is_integral_v<T>)
    {
      store(value.id);
      return *this;
    }

  private:
    int ensure_binding() const
    {
      auto &ast = vlk::ast();
      if (owner->binding_ < 0) { owner->binding_ = ast.next_binding++; }
      for (auto &existing : ast.buffers) {
        if (existing.binding == owner->binding_) {
          existing.vk_buffer = owner->buffer_;
          existing.byte_size = owner->count_ * sizeof(T);
          existing.elem_count = owner->count_;
          return owner->binding_;
        }
      }
      vlk::BufferBinding bb;
      bb.name = owner->name_;
      bb.elem_glsl_type = std::is_floating_point_v<T> ? "float" : "int";
      bb.binding = owner->binding_;
      bb.vk_buffer = owner->buffer_;
      bb.byte_size = owner->count_ * sizeof(T);
      bb.elem_count = owner->count_;
      ast.buffers.push_back(bb);
      if (ast.next_binding <= owner->binding_) { ast.next_binding = owner->binding_ + 1; }
      return owner->binding_;
    }

    vlk::Float load_float() const
    {
      const int binding = ensure_binding();
      vlk::ExprNode n = vlk::ExprNode::make(vlk::OpKind::Load, index.id);
      n.binding = binding;
      const int load = vlk::ast().append(std::move(n));
      const int tmp = vlk::ast().make_temp("f");
      vlk::emit_assign(tmp, load);
      return vlk::Float{ tmp };
    }
    vlk::Int load_int() const
    {
      const int binding = ensure_binding();
      vlk::ExprNode n = vlk::ExprNode::make(vlk::OpKind::Load, index.id);
      n.binding = binding;
      const int load = vlk::ast().append(std::move(n));
      const int tmp = vlk::ast().make_temp("i");
      vlk::emit_assign(tmp, load);
      return vlk::Int{ tmp };
    }
    void store(int value_id) const
    {
      const int binding = ensure_binding();
      vlk::ExprNode n = vlk::ExprNode::make(vlk::OpKind::Store, index.id, value_id);
      n.binding = binding;
      vlk::ast().append(std::move(n));
    }
  };

  Ref operator[](vlk::Int idx) { return Ref{ this, idx }; }

  void register_for_dispatch(vlk::ASTContext &ast) const
  {
    if (binding_ < 0) { return; }
    for (auto &b : ast.buffers) {
      if (b.binding == binding_) {
        b.vk_buffer = buffer_;
        b.byte_size = count_ * sizeof(T);
        b.elem_count = count_;
      }
    }
  }

private:
  static int next_name_id()
  {
    static int id = 0;
    return id++;
  }

  std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags props) const
  {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx_->physical_device(), &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) { return i; }
    }
    throw std::runtime_error("no suitable host-visible memory type");
  }

  context *ctx_{ nullptr };
  VkBuffer buffer_{ VK_NULL_HANDLE };
  VkDeviceMemory memory_{ VK_NULL_HANDLE };
  void *mapped_{ nullptr };
  std::size_t count_{ 0 };
  std::string name_;
  mutable int binding_{ -1 };
};

} // namespace vkexec
