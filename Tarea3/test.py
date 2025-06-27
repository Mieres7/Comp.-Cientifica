def build_up_b(rho, dt, u, v, dx, dy):
    """
    Construye el término del lado derecho de la ecuación de Poisson para la presión a partir de los campos de velocidad u y v.

    Parameters:
     - rho : Densidad del fluido.
     - dt : Paso temporal.
     - u : Componente horizontal de la velocidad.
     - v : Componente vertical de la velocidad.
     - dx : Tamaño de celda en x.
     - dy : Tamaño de celda en y.
    Returns:
     - b : Matriz del término fuente para la ecuación de Poisson.
    """
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * ((1/dt) * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*dx) +
                                     (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*dy)) -
                         ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*dx))**2 -
                         2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2*dy) *
                              (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2*dx)) -
                         ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*dy))**2)

    return b


def pressure_poisson(p, b, dx, dy, nit=50):
    """
    Resuelve la ecuación de Poisson para el campo de presión `p` utilizando un método iterativo

    Parameters:
    p : Campo inicial de presión.
    b : Lado derecho de la Ecuación.
    dx : Tamaño de celda en x.
    dy : Tamaño de celda en y.
    nit : Número de iteraciones para el método iterativo (default 50).

    Returns:
    p : ndarray
        Campo de presión actualizado.
    """
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
             dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1]) -
             b[1:-1, 1:-1] * dx**2 * dy**2)
            / (2 * (dx**2 + dy**2))
        )

        p[:, -1] = p[:, -2]  # salida
        p[:, 0] = p[:, 1]    # entrada
        p[0, :] = p[1, :]    # pared inferior
        p[-1, :] = p[-2, :]  # pared superior
    return p


def simulation(a, b, d, u0, nu, nx = 41, ny = 41, nt=100, better=False):

    """
    Ejecuta una simulación bidimensional del flujo de un fluido incompresible alrededor de un obstáculo circular en un canal rectangular

    Parameters:
    a : Altura del canal.
    b : Ancho del canal.
    d : Distancia del centro del obstáculo desde el borde izquierdo.
    u0 : Velocidad de entrada y de las paredes móviles.
    nu : Viscosidad cinemática del fluido.
    nx : Número de puntos en la dirección x (default = 41).
    ny : Número de puntos en la dirección y (default = 41).
    nt : Número de iteraciones.

    Returns:
    u : Componente horizontal del campo de velocidad.
    v : Componente vertical del campo de velocidad.
    p : Campo de presión.
    X : Malla de coordenadas X para visualización.
    Y : Malla de coordenadas Y para visualización.
    stepcount : Número de iteraciones realizadas hasta la convergencia.
    """

    rho = 1   # densidad
    dx = b / (nx - 1)
    dy = a / (ny - 1)
    dt = 0.01  # paso temporal
    F = 1


    # Dimensiones del obstáculo
    r = a / 25
    xc = d
    yc = a / 2

    x = np.linspace(0, b, nx)
    y = np.linspace(0, a, ny)

    X, Y = np.meshgrid(x, y)

    obstacle_mask = (X - xc)**2 + (Y - yc)**2 <= r**2
    obstacle_edge = binary_dilation(obstacle_mask) & (~obstacle_mask) # Obtención del contorno

    # Inicialización de vectores
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    udiff = 0
    stepcount = 0

    forces = []
    states = []

    while udiff < nt:

        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, b, dx, dy)

        force_t = np.sum(p[obstacle_edge]) * dx * dy  # Integral de presión en el borde -> Calculo de fueraza.
        forces.append(force_t)

        # Actualizar las velocidades utilizando la fórmula de Navier-Stokes
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                        dt / (2 * rho * dx) *
                        (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                        F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
            dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
            nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Incorporacion del Obstáculo
        u[obstacle_mask] = 0
        v[obstacle_mask] = 0

        # Condiciones de borde
        u[:, 0]  = u0       # entrada
        u[:, -1] = u[:, -2] # salida
        v[:, 0]  = 0
        v[:, -1] = 0

        u[0, :]  = u0       # pared inferior
        u[-1, :] = u0       # pared superior
        v[0, :]  = 0
        v[-1, :] = 0

        # udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        udiff = udiff + 1
        stepcount += 1
        states.append((u.copy(), v.copy(), p.copy())) # linea solo necesaria para la animacion

    return u, v, p, X, Y, stepcount, forces, states, obstacle_mask, xc, yc, r
