#include <cstdlib>
#include <fstream>
#include <string>
int main(int argc, char **argv)
{
  if (char const *path = std::getenv("AUDIT_LOG")) {
    std::ofstream log(path, std::ios::app);
    log << "baseline";
    for (int i = 1; i < argc; ++i) log << "|" << argv[i];
    log << "\n";
  }
  char const *failure = std::getenv("FAIL_STAGE");
  return failure && std::string(failure) == "baseline" ? 1 : 0;
}
