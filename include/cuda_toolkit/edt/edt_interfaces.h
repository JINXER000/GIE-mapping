#ifndef EDT_INTERFACES_H
#define EDT_INTERFACES_H

#include <cuda_toolkit/projection.h>
#include <cutt/cutt.h>
#include "map_structure/local_batch.h"

namespace EDT_OCC
{
    void batchEDTUpdate(LocMap *loc_map, cuttHandle *plan, const int time);
}

#endif // EDT_INTERFACES_H
