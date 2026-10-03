#pragma once

#include "sceneStructs.h"

namespace Compaction
{
    void init(int maxN);
    void free();

    // stable partition of paths[0, n) in place. dead paths paths[n, ...) aren't touched so paths that
    // died on earlier bounces are still there for finalGather. returns the number alive
    int partitionPaths(int n, PathSegment* paths);
}
