#!/usr/bin/env bash
set -euo pipefail

expected_version='shaderc v2026.1'
compiler_version="$(glslc --version)"
if [[ "${compiler_version%%$'\n'*}" != "$expected_version" ]]; then
  echo "Expected $expected_version to validate committed sample shaders" >&2
  exit 1
fi

shader_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT

while IFS= read -r -d '' shader; do
  relative_path="${shader#"$shader_root"/}"
  output="$temporary_dir/$relative_path.spv"
  mkdir -p "$(dirname "$output")"
  glslc --target-env=vulkan1.3 -I "$(dirname "$shader")" "$shader" -o "$output"
  if ! cmp -s "$output" "$shader.spv"; then
    echo "Committed SPIR-V differs from $relative_path" >&2
    exit 1
  fi
done < <(find "$shader_root" -type f \( -name '*.vert' -o -name '*.frag' -o -name '*.comp' \) -print0)
