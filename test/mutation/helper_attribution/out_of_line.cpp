#include "validity.hpp"
bool valid(image_use const &use) noexcept
{
  switch (use.kind) {
  case usage::sampled:
    return use.mode == access::read;
  case usage::transfer_destination:
    return use.mode == access::write;
  case usage::storage:
    return true;
  }
  return false;
}
