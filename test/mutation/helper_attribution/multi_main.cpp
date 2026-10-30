#include "validity.hpp"
bool (*volatile negative_control)(image_use const &) noexcept = unexecuted;
int a(int);
int b(int);
int main(int argc, char **) { return a(argc) + b(argc + 1); }
