#include <vkexec/error_helpers.hpp>

#include <vkexec/error.hpp>

#include <boost/leaf/error.hpp>

#include <vulkan/vulkan_core.h>

#include <string>
#include <string_view>

namespace vkexec {

auto make_vk_error(VkResult result, std::string_view context) -> leaf::error_id
{
  return detail::stash_error(error{
    .code = make_vk_error_code(static_cast<int>(result)),
    .detail = std::string(context),
  });
}

}// namespace vkexec
