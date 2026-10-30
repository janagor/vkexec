#pragma once

enum class access { read, write, read_write };
enum class usage { sampled, transfer_destination, storage };
struct image_use
{
  usage kind;
  access mode;
};

#ifdef ATTRIBUTION_ORDINARY
#define ATTRIBUTION_FUNCTION inline
#else
#define ATTRIBUTION_FUNCTION constexpr
#endif

#ifdef ATTRIBUTION_OUT_OF_LINE
bool valid(image_use const &use) noexcept;
#else
ATTRIBUTION_FUNCTION bool valid(image_use const &use) noexcept
{
#ifdef ATTRIBUTION_IF
  if (use.kind == usage::sampled) return use.mode == access::read;
  if (use.kind == usage::transfer_destination) return use.mode == access::write;
  return true;
#else
  switch (use.kind) {
  case usage::sampled:
    return use.mode == access::read;
  case usage::transfer_destination:
    return use.mode == access::write;
  case usage::storage:
    return true;
  }
  return false;
#endif
}
#endif

// Emitted and retained through a volatile function pointer, but never called.
inline bool unexecuted(image_use const &use) noexcept { return use.mode == access::read; }

inline int exercise(int seed)
{
  // argc supplies a runtime offset; all nine combinations execute in every run.
  for (int i = 0; i < 3; ++i) {
    auto mode = static_cast<access>((seed + i) % 3);
    if (valid({ usage::sampled, mode }) != (mode == access::read)) return 1;
    if (valid({ usage::transfer_destination, mode }) != (mode == access::write)) return 2;
    if (!valid({ usage::storage, mode })) return 3;
  }
  return 0;
}
