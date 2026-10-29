#pragma once

// Must stay uncovered in both profile modes; detects accidental filter bypass.
__attribute__((used)) inline bool uncalled(int a, int b) { return a < b; }

#ifdef ATTRIBUTION_OUT_OF_LINE
bool overlaps(int a0, int a1, int b0, int b1);
#else
inline bool overlaps(int a0, int a1, int b0, int b1) { return a0 < b1 && b0 < a1; }
#endif

template<typename T> bool template_overlaps(T a0, T a1, T b0, T b1) { return a0 < b1 && b0 < a1; }

inline bool exercise()
{
  return overlaps(0, 4, 2, 6) && !overlaps(0, 4, 4, 6) && !overlaps(0, 2, 4, 6) && template_overlaps<int>(0, 4, 2, 6)
         && !template_overlaps<int>(0, 4, 4, 6) && !template_overlaps<int>(0, 2, 4, 6)
         && template_overlaps<long>(0, 4, 2, 6) && !template_overlaps<long>(0, 4, 4, 6)
         && !template_overlaps<long>(0, 2, 4, 6);
}
