
from .memory_strategy.global_lt_balance_update import Global_lt_balance_update


from .memory_strategy.part_st_balance_retrieve import Part_st_balance_retrieve
from .memory_strategy.part_st_balance_update import Part_st_balance_update
from .memory_strategy.part_lt_balance_retrieve import Part_lt_balance_retrieve
from .memory_strategy.part_lt_balance_update import Part_lt_balance_update

from .memory_strategy.global_st_balance_update import Global_st_balance_update
from .memory_strategy.global_st_balance_retrieve import Global_st_balance_retrieve
from .memory_strategy.global_lt_balance_retrieve import Global_lt_balance_retrieve

from .memory_strategy.global_lt_reservoir_update import Global_lt_reservoir_update
from .memory_strategy.part_lt_reservoir_update import Part_lt_reservoir_update



retrieve_methods = {
    'global_st_balance': Global_st_balance_retrieve,  # retrieve all samples
    'global_lt_balance':Global_lt_balance_retrieve,  # half-pos and half-neg

    'part_st_balance': Part_st_balance_retrieve,  # retrieve all samples
    'part_lt_balance': Part_lt_balance_retrieve,  # half-pos and half-neg
}

update_methods = {
    'global_st_balance': Global_st_balance_update,
    'global_lt_balance': Global_lt_balance_update,
    'global_lt_reservoir': Global_lt_reservoir_update,

    'part_st_balance': Part_st_balance_update,
    'part_lt_balance': Part_lt_balance_update,
    'part_lt_reservoir': Part_lt_reservoir_update,

}