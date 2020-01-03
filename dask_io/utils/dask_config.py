import dask 


def manual_config_dask(buffer_size=ONE_GIG, opti=True, sched_opti=True):
    """ Manual configuration of Dask as opposed to using CaseConfig object.

    Arguments:
    ----------
        buffer_size:
        opti:
        sched_opti:
    """

    print('Task graph optimization enabled:', opti)
    print('Scheduler optimization enabled:', sched_opti)

    opti_funcs = [optimize_func] if opti else list()
    dask.config.set({
        'optimizations': opti_funcs,
        'io-optimizer': {
            'memory_available': buffer_size,
            'scheduler_opti': sched_opti
        }
    })


def configure_dask(config):
    """ Apply configuration to dask to parameterize the optimization function.

    Arguments:
    ---------
        config: A CaseConfig object.
    """
    if not config:
        raise ValueError("Empty configuration object.")
    manual_config_dask(config.buffer_size, config.opti, sched_opti=scheduler_opti)