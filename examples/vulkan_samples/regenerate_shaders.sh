#!/usr/bin/env bash
set -euo pipefail

expected_version='shaderc v2026.1'
compiler_version="$(glslc --version)"
if [[ "${compiler_version%%$'\n'*}" != "$expected_version" ]]; then
  echo "Expected $expected_version to regenerate committed sample shaders" >&2
  exit 1
fi

shader_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while IFS= read -r -d '' shader; do
  glslc --target-env=vulkan1.3 -I "$(dirname "$shader")" "$shader" -o "$shader.spv"
done < <(find "$shader_root" -type f \( -name '*.vert' -o -name '*.frag' -o -name '*.comp' \) -print0)
