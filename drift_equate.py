from __future__ import print_function, division
from ciabatta import ejm_rcparams
import matplotlib.pyplot as plt
import numpy as np


def seg_intersect(p1, p2, yi):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    xi = (yi - c) / m
    if x1 < xi < x2:
        return xi
    else:
        raise ValueError


def curve_intersect(xs, ys, yi):
    i_big = np.where(ys > yi)[0][0]
    p1 = [xs[i_big - 1], ys[i_big - 1]]
    p2 = [xs[i_big], ys[i_big]]
    return seg_intersect(p1, p2, yi)


vd_0 = 0.25

d = np.load('free_space_drift_curve.npz')
(chi_r_s, vd_r_s, vde_r_s,
 chi_a_s, vd_a_s, vde_a_s,
 chi_r_t, vd_r_t, vde_r_t,
 chi_a_t, vd_a_t, vde_a_t) = (d['chi_r_s'], d['vd_r_s'], d['vde_r_s'],
                              d['chi_a_s'], d['vd_a_s'], d['vde_a_s'],
                              d['chi_r_t'], d['vd_r_t'], d['vde_r_t'],
                              d['chi_a_t'], d['vd_a_t'], d['vde_a_t'])

plt.axhline(vd_0)
for i, vset in enumerate(([chi_r_s, vd_r_s, vde_r_s],
                          [chi_a_s, vd_a_s, vde_a_s],
                          [chi_r_t, vd_r_t, vde_r_t],
                          [chi_a_t, vd_a_t, vde_a_t])):
    chi, vd, vde = vset
    plt.errorbar(chi, vd, yerr=vde, c=ejm_rcparams.set2[i])
    chi_0 = curve_intersect(chi, vd, vd_0)
    plt.axvline(chi_0, c=ejm_rcparams.set2[i])
    print(chi_0)

plt.show()
