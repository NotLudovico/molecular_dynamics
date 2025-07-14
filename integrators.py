def force(q):
    return -q


dt = 0.01
m = 2


def verlet():
    q_0 = 0
    q = 0

    for i in range(1000):
        q_next = 2 * q - q_0 + force(q) * dt**2 / m
        q_0 = q
        q = q_next


def velocity_verlet():
    q = 0
    p = 0
    f = force(q)

    for i in range(1000):
        p += force(q) * dt / 2
        q += p / m * dt
        f = force(q)  # Update force each time the position is updated
        p += f * dt / 2


def leapfrog():
    q = 0
    p = 0
    f = force(q)

    p += p + f * dt / 2
    for i in range(1000):
        q += p / m * dt
        f = force(q)
        p += f * dt


def position_verlet():
    q = 0
    p = 0
    f = force(q)

    for i in range(1000):
        q += p / m * dt**2
        f = force(q)
        p += f * dt
        q += p / m * dt**2


def forcefast(q):
    return -10 * q


def forceslow(q):
    return -q


def wrong_multiple_ts():
    q = 0
    p = 0
    f1 = forcefast(q)
    f2 = forceslow(q)
    f = f1 + f2

    for i in range(1000):
        p += f * dt / 2
        q += p / m * dt
        f1 = forcefast(q)
        if i % 5 == 0:
            f2 = forceslow(q)

        f = f1 + f2
        p += f * dt / 2


def multiple_ts():
    q = 0
    p = 0
    ddt = dt / 10
    f1 = forcefast(q)
    f2 = forceslow(q)

    for i in range(1000):
        p += f1 * ddt / 2
        if i % 10 == 0:
            p += f2 * dt / 2
        q += p / m * ddt
        f1 = forcefast(q)
        p += f1 * ddt / 2

        if i % 10 == 0:
            f2 = forceslow(q)
            p += f2 * dt / 2
