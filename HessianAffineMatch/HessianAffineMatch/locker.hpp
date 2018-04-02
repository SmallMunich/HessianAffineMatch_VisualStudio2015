#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <stdlib.h>

using namespace std;

namespace Locker {
  bool lock(const string &fpath) {
    struct stat info;
    string lock_fpath = fpath + ".lock";
    if (stat(fpath.c_str(), &info) != 0 && 
        stat(lock_fpath.c_str(), &info) != 0) {
      //mkdir(lock_fpath.c_str(),
      //    S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      system((string("mkdir -p ") + lock_fpath).c_str());
      return true;
    } else {
      return false;
    }
  }

  void unlock(const string &fpath) {
    string lock_fpath = fpath + ".lock";
    //rmdir(lock_fpath.c_str());
	remove(lock_fpath.c_str());
  }
}
