#ifndef VKEXEC_PIPELINE_CACHE_HPP
#define VKEXEC_PIPELINE_CACHE_HPP

#include <vkexec/pipeline.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>

namespace vkexec::edsl {
struct ASTContext;
}

namespace vkexec {

class context;

class pipeline_cache
{
public:
  explicit pipeline_cache(context &ctx);
  ~pipeline_cache();

  pipeline_cache(pipeline_cache const &) = delete;
  auto operator=(pipeline_cache const &) -> pipeline_cache & = delete;
  pipeline_cache(pipeline_cache &&) = delete;
  auto operator=(pipeline_cache &&) -> pipeline_cache & = delete;

  auto get_or_compile(edsl::ASTContext const &ast, std::uint32_t work_count) -> pipeline_resources &;
  auto get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
    -> pipeline_resources &;

private:
  context *ctx_;
  std::mutex mutex_;
  std::unordered_map<std::size_t, std::unique_ptr<pipeline_resources>> cache_;
};

}// namespace vkexec

#endif// VKEXEC_PIPELINE_CACHE_HPP
