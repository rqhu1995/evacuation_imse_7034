# -*- coding:utf-8 _*-
""" 
@author:Runqiu Hu
@license: Apache Licence 
@file: evacuation.py 
@time: 2021/11/22
@contact: hurunqiu@live.com
@project: evacuation

"""
import gurobipy as gp
import numpy as np
from gurobipy import GRB

demand_type = [1, 2, 3]
tolerate_time = {1: 2, 2: 6, 3: 12}
max_cycle = {
    1: [3, 7, 14],
    2: [4, 8, 16],
    3: [4, 8, 16],
    4: [4, 8, 16],
    5: [3, 7, 14],
}

speed = {
    1: 240, 2: 280, 3: 280, 4: 280, 5: 240
}

capacity = {
    1: 25, 2: 12, 3: 12, 4: 12, 5: 5
}

base = {
    1: 8, 2: 9, 3: 9, 4: 9, 5: 8
}

mdl = gp.Model("evacuation")
base_node = [8, 9]
demand_node = [4, 5, 6, 7]
relief_node = [1, 2, 3]
light_helicopter = [1, 5]
heavy_helicopter = [2, 3, 4]
helicopters = light_helicopter + heavy_helicopter
all_nodes = relief_node + demand_node + base_node

demand = {
    (4, 1): 15,
    (4, 2): 61,
    (4, 3): 20,
    (5, 1): 25,
    (5, 2): 42,
    (5, 3): 112,
    (6, 1): 77,
    (6, 2): 16,
    (6, 3): 33,
    (7, 1): 92,
    (7, 2): 42,
    (7, 3): 13,
}

endurance = {
    1: 3, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2
}

# region dist_matrix
dist_matrix = np.array([[0., 16.02132036, 40.05850649, 24.07551559, 58.79354933,
                         38.86104255, 51.94094342, 14.01183102, 27.63265424],
                        [16.02132036, 0., 41.18179723, 28.26855294, 54.60656798,
                         22.91672529, 40.25142526, 28.98663962, 37.95951348],
                        [40.05850649, 41.18179723, 0., 63.89736639, 22.82681806,
                         49.82618183, 37.49948425, 49.89262866, 23.91730563],
                        [24.07551559, 28.26855294, 63.89736639, 0., 81.17339692,
                         46.26371343, 68.36533401, 21.51538915, 50.42907071],
                        [58.79354933, 54.60656798, 22.82681806, 81.17339692, 0.,
                         53.67533163, 29.82180894, 70.52153972, 46.74293044],
                        [38.86104255, 22.91672529, 49.82618183, 46.26371343, 53.67533163,
                         0., 28.27025191, 51.81221785, 56.11760102],
                        [51.94094342, 40.25142526, 37.49948425, 68.36533401, 29.82180894,
                         28.27025191, 0., 65.8862377, 55.18393322],
                        [14.01183102, 28.98663962, 49.89262866, 21.51538915, 70.52153972,
                         51.81221785, 65.8862377, 0., 31.42439949],
                        [27.63265424, 37.95951348, 23.91730563, 50.42907071, 46.74293044,
                         56.11760102, 55.18393322, 31.42439949, 0.]])
# endregion

node_node_heli_loop_dtype = [(i, j, k, m, n) for i in all_nodes for j in all_nodes
                             for k in helicopters for m in demand_type
                             for n in range(1, max_cycle[k][m - 1] + 1) if i != j]
node_heli_loop_dtype = [(i, k, m, n) for i in all_nodes
                             for k in helicopters for m in demand_type
                             for n in range(1, max_cycle[k][m - 1] + 1)]
node_dtype = [(i, m) for i in demand_node for m in demand_type]
dtype_heli = [(m, k) for m in demand_type for k in helicopters]
heli_dtype_loop = [(k, m, n) for k in helicopters for m in demand_type for n in range(1, max_cycle[k][m - 1] + 1)]

x = mdl.addVars(node_node_heli_loop_dtype, vtype=GRB.BINARY, name="visit")
W = mdl.addVars(node_dtype, vtype=GRB.INTEGER, name="unserved")
E = mdl.addVars([(i, 0) for i in demand_node] + node_dtype, vtype=GRB.INTEGER, name="overserved")
actual_cycle = mdl.addVars(dtype_heli, vtype=GRB.INTEGER, name="actual_cycles")
y_auxi = mdl.addVars(node_dtype, vtype=GRB.BINARY, name="auxiliary_y")
z_auxi = mdl.addVars(node_dtype, vtype=GRB.BINARY, name="auxiliary_z")
u_auxi = mdl.addVars(heli_dtype_loop, vtype=GRB.BINARY, name="auxiliary_u")
urgent_limit_time = {
    1: 2,
    2: 6,
    3: 12
}

v_mtz_auxiliary = mdl.addVars(node_heli_loop_dtype, vtype=GRB.INTEGER, lb=0, name="mtz")
accu_m_time = mdl.addVars(dtype_heli, vtype=GRB.CONTINUOUS, name="accum_time")

mdl.addConstrs(E[i, 0] == 0 for i in demand_node)
mdl.addConstrs(
    W[i, m] - E[i, m] ==
    demand[(i, m)] - E[i, m - 1] - gp.quicksum(
        capacity[k] * x[j, i, k, m, n] for k in helicopters for j in base_node + relief_node
        for n in range(1, max_cycle[k][m - 1] + 1)
    )
    for i in demand_node for m in demand_type
)

# 一个loop从base开始
mdl.addConstrs(
    gp.quicksum(
        x[base[k], j, k, m, n]
        for j in demand_node
    ) <= 1
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(
        x[base[k], j, k, m, n]
        for j in relief_node
    ) == 0
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

# 一个base结束
mdl.addConstrs(
    gp.quicksum(
        x[j, base[k], k, m, n]
        for j in relief_node
    ) <= 1
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(
        x[j, base[k], k, m, n]
        for j in demand_node
    ) == 0
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(
        x[j, base[k], k, m, n]
        for j in relief_node
    ) == gp.quicksum(
        x[base[k], j, k, m, n]
        for j in demand_node
    )
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(
        x[i, j, k, m, n]
        for i in all_nodes
        for j in all_nodes
        if i != j
    ) <=
    10000 * gp.quicksum(
        x[base[k], j, k, m, n]
        for j in demand_node
    )
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    y_auxi[i, m] + z_auxi[i, m] == 1
    for i in demand_node for m in demand_type
)

mdl.addConstrs(
    W[i, m] <= 10000 * y_auxi[i, m]
    for i in demand_node for m in demand_type
)

mdl.addConstrs(
    E[i, m] <= 10000 * z_auxi[i, m]
    for i in demand_node for m in demand_type
)

mdl.addConstrs(
    gp.quicksum(
        x[i, j, k, m, n]
        for i in all_nodes
        if j != i
    ) ==
    gp.quicksum(
        x[j, i, k, m, n]
        for i in all_nodes
        if j != i
    )
    for j in all_nodes
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    x[i, j, k, m, n] == 0
    for i in base_node
    for j in base_node
    if j != i
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(x[i, j, k, m, n] for i in base_node for j in all_nodes if j != i) <= 1
    for k in helicopters for m in demand_type for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(x[i, j, k, m, n] for i in all_nodes if j != i) <= 1
    for j in all_nodes for k in helicopters for m in demand_type for n in range(1, max_cycle[k][m - 1] + 1)
)


mdl.addConstrs(
    constrs=(
        v_mtz_auxiliary[i, k, m, n] - v_mtz_auxiliary[j, k, m, n] + 1 <= (1 - x[i, j, k, m, n]) * 10
        for i in relief_node + demand_node
        for j in relief_node + demand_node
        for k in helicopters if i != j
        for m in demand_type
        for n in range(1, max_cycle[k][m - 1] + 1)
    ), name="sub_tour_elimination"
)

# time constraints
# 一个loop不能超过飞机最长飞行时间
mdl.addConstrs(
    gp.quicksum(
        x[i, j, k, m, n] * dist_matrix[i - 1][j - 1] / speed[k]
        for i in all_nodes for j in all_nodes if i != j
    ) <= endurance[k]
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    actual_cycle[m, k] == gp.quicksum(
        x[base[k], j, k, m, n] for j in all_nodes if j != base[k] for n in range(1, max_cycle[k][m - 1] + 1)
    ) for k in helicopters for m in demand_type
)

mdl.addConstrs(
    gp.quicksum(
        x[i, j, k, m, n] for j in base_node + demand_node if j != i
    ) == 0
    for i in demand_node
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    gp.quicksum(
        x[i, j, k, m, n] for j in relief_node if j != i
    ) == 0
    for i in relief_node
    for k in helicopters
    for m in demand_type
    for n in range(1, max_cycle[k][m - 1] + 1)
)

mdl.addConstrs(
    accu_m_time[m, k] == (
            gp.quicksum(
                dist_matrix[i - 1, j - 1] / speed[k] * x[i, j, k, m, n]
                for i in all_nodes
                for j in all_nodes
                for n in range(1, max_cycle[k][m - 1] + 1)
                if j != i
            ) + (actual_cycle[m, k] - 1) * 0.5
    )
    for k in helicopters
    for m in demand_type

)

mdl.addConstrs(
    gp.quicksum(
        u_auxi[k, m, n] for n in range(1, max_cycle[k][m - 1] + 1)
    ) == 1
    for k in helicopters
    for m in demand_type
)

mdl.addConstrs(
    gp.quicksum(
        n * u_auxi[k, m, n] for n in range(1, max_cycle[k][m - 1] + 1)
    ) == actual_cycle[m, k]
    for k in helicopters
    for m in demand_type
)

mdl.addConstrs(
    gp.quicksum(
        accu_m_time[mc, k] for mc in range(1, m + 1)
    ) -
    gp.quicksum(
        x[i, base[k], k, m, n] * u_auxi[k, m, n] * dist_matrix[i - 1, base[k] - 1] / speed[k]
        for i in all_nodes if i != base[k]
        for n in range(1, max_cycle[k][m - 1] + 1)
    ) <= urgent_limit_time[m]
    for m in demand_type
    for k in helicopters
)

mdl.setObjective(
    gp.quicksum(
        W[i, m]
        for i in demand_node
        for m in demand_type
    ), GRB.MINIMIZE
)

mdl.optimize()

for v in mdl.getVars():
    if v.varName.startswith("visit") and \
            v.varName[-6] == "5" and \
            v.varName[-4] == "3" and \
            v.varName[-2] == "14" and \
            v.x != 0:
        print(v.varName, v.x)
