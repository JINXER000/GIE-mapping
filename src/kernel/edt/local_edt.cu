
#include "cuda_toolkit/edt/edt_interfaces.h"
#include "local_edt_core.h"

namespace EDT_OCC
{
    void batchEDTUpdate(LocMap *loc_map, cuttHandle *plan, const int time)
    {

        //Phase 1
        EDT_CORE::EDTphase1<<<loc_map->_local_size.z,loc_map->_local_size.x>>>(*loc_map,time);
        cuttExecute(plan[0], loc_map->_g, loc_map->_aux);
        cuttExecute(plan[0], loc_map->_coc_idx, loc_map->_coc_idx_aux);

        //Phase 2
        EDT_CORE::EDTphase2<<<loc_map->_local_size.z,loc_map->_local_size.y>>>(*loc_map);
        cuttExecute(plan[1], loc_map->_g, loc_map->_aux);
        cuttExecute(plan[1], loc_map->_coc_idx, loc_map->_coc_idx_aux);

        //Phase 3 (needed only for 3D map)
        if (loc_map->_local_size.z > 1)
        {
            EDT_CORE::EDTphase3<<<loc_map->_local_size.x,loc_map->_local_size.y>>>(*loc_map);
            cuttExecute(plan[2], loc_map->_g, loc_map->_aux);
            cuttExecute(plan[2], loc_map->_coc_idx, loc_map->_coc_idx_aux);
        }

    }
}