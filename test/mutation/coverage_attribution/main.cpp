#include "ranges.hpp"
#ifdef ATTRIBUTION_MULTI_TU
bool a();
bool b();
#endif
int main()
{
#ifdef ATTRIBUTION_MULTI_TU
  return a() && b() ? 0 : 1;
#else
  return exercise() ? 0 : 1;
#endif
}
