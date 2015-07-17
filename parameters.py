defaults = {
    'L': 1.0,
    'dim': 2,
    'dt': 0.001,
    'rho_0': 1000.0,
    'v_0': 1.0,
    'D_rot_0': 1.0,
    'chi': None,
    'seed': 1,
    'rc': None,
    'Rc': 0.05,
    'dx': None,
    'food_0': None,
    'gamma': None,
    'D_food': None,
}

agent_defaults = {
    'dt_chemo': defaults['dt'],
    'memory': None,
    't_mem': 5.0,
    'fixed_food_gradient': False,
}
