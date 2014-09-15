import fipy
import numpy as np


gmsh_text_box = '''
dx = %(dx)g;
R = %(R)g;
Lx = %(Lx)g;
Ly = %(Ly)g;

// define the corners of the square
Point(1) = {Lx / 2, Ly / 2, 0, dx / 2};
Point(2) = {-Lx / 2, Ly / 2, 0, dx / 2};
Point(3) = {-Lx / 2, -Ly / 2, 0, dx / 2};
Point(4) = {Lx / 2, -Ly / 2, 0, dx / 2};

// define the square
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};

Line Loop(1) = {1, 2, 3, 4};

'''

gmsh_text_circle = '''
x = %(x)g;
y = %(y)g;
i = %(i)d;

// define the compass points of the circle
Point(i) = {x, y, 0, dx};
Point(i + 1) = {x - R, y, 0, dx};
Point(i + 2) = {x, y + R, 0, dx};
Point(i + 3) = {x + R, y, 0, dx};
Point(i + 4) = {x, y - R, 0, dx};

// define the circle
Circle(i) = {i + 1, i, i + 2};
Circle(i + 1) = {i + 2, i, i + 3};
Circle(i + 2) = {i + 3, i, i + 4};
Circle(i + 3) = {i + 4, i, i + 1};

// define the boundaries
Line Loop(i) = {i, i + 1, i + 2, i + 3};

'''


def make_porous_mesh(r, R, dx, L):
    gmsh_text = gmsh_text_box % {'dx': dx[0], 'R': R,
                                 'Lx': L[0], 'Ly': L[1]}
    for i in range(len(r)):
        index = 5 * (i + 1)
        gmsh_text += gmsh_text_circle % {'x': r[i][0], 'y': r[i][1],
                                         'i': index}
    loops = ', '.join([str(n) for n in range(5, 5 * (len(r) + 1), 5)])
    gmsh_text += 'Plane Surface(1) = {1, %s};' % loops
    return fipy.Gmsh2D(gmsh_text)

if __name__ == '__main__':
    r = np.array([[0.0, 0.1], [0.3, 0.3]])
    R = 0.1
    dx = np.array([0.02, 0.02])
    L = np.array([1.0, 1.0])
    m = make_porous_mesh(r, R, dx, L)

    phi = fipy.CellVariable(m)
    v = fipy.Viewer(vars=phi, xmin=-L[0] / 2.0, xmax=L[0] / 2.0)
    v.plotMesh()
    raw_input()
