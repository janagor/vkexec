#include "validity.hpp"
bool (*volatile negative_control)(image_use const &) noexcept = unexecuted;
int main(int argc, char **) { return exercise(argc); }
