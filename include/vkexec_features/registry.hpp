#ifndef VKEXEC_FEATURES_REGISTRY_HPP
#define VKEXEC_FEATURES_REGISTRY_HPP

/**
 * Central list of vkexec registered features.
 *
 * Expand with `VKEXEC_FEATURE_LIST(X)` where `X(name)` is invoked once per feature tag.
 */
#define VKEXEC_FEATURE_LIST(X) \
  X(timeline_semaphore)        \
  X(buffer_device_address)     \
  X(dynamic_rendering)

#endif// VKEXEC_FEATURES_REGISTRY_HPP
