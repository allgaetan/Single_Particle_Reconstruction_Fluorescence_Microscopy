import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.widgets import Slider, Button
from time import time


def get_rotation_mat(angle):
    angle = float(angle)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])


def simu_cylinder_surface_random(a, b, l, N):
    angle = np.linspace(0, 2*np.pi, num=N, endpoint=False)
    x = a * np.cos(angle)
    y = b * np.sin(angle)
    z = l * np.random.random(size=(N,))
    #z = l * np.linspace(0, 1, num=N)
    return np.stack([x, y, z], axis=1)


def simu_triplet_random(r, l, s, nbPts):
    n12 = nbPts // 3
    n3 = nbPts - 2 * n12
    V1 = simu_cylinder_surface_random(r,r,l,n12)
    V2 = simu_cylinder_surface_random(r,r,l,n12)
    V3 = simu_cylinder_surface_random(r,r,l,n3)
    V2[:,0] += s
    V3[:,0] += 2*s
    return np.concatenate([V1,V2,V3], axis=0)


def simu_cylinder_rev(radius, r, l, s, nbPts, angle, nb_missing_triplets=0, triplet=1):
    Cn = 9
    angle = angle - 180
    #t0 = time()
    # Radius variation profile
    x0 = np.linspace(0, l, num=len(radius))
    y0 = radius
    radius_profile_spl = UnivariateSpline(x0, y0, s=1)
    #t1 = time()

    samplingThetaCn = 2 * np.pi / Cn
    if triplet == 1:
        Vi0 = simu_triplet_random(r, l, s, nbPts)
    else:
        Vi0 = simu_cylinder_surface_random(r*1, r*0.2, l, nbPts)
    #t2 = time()
    angle = angle * np.pi / 180
    R = get_rotation_mat(angle)
    Vi0 = (R @ Vi0[:,:,None])[:,:,0]
    V = np.zeros((nbPts, Cn-nb_missing_triplets, 3))
    missing_triplets = range(9)[:nb_missing_triplets]
    non_missing_angles = [i*samplingThetaCn for i in range(Cn) if i not in missing_triplets]
    #t3 = time()
    for i in range(len(non_missing_angles)):
        angle = non_missing_angles[i]
        R = get_rotation_mat(angle)
        V[:, i] = (R @ Vi0[:,:,None])[:,:,0]
        Dinner = radius_profile_spl(V[:, i, 2])
        V[:, i, 0] += (Dinner + r) * np.cos(angle)
        V[:, i, 1] += (Dinner + r) * np.sin(angle)
    #t4 = time()

    #print(f"Spline calc : {t1-t0:.6f}s")
    #print(f"Vi0 calc : {t2-t1:.6f}s")
    #print(f"Rotation : {t3-t2:.6f}s")
    #print(f"V calc : {t4-t3:.6f}s")
    return V.reshape(-1, 3)


def get_radius(x, a, b, c):
    # get the radius
    return np.round(a * (x - b) ** 2  + c)


if __name__ == '__main__':
    np.random.seed(0)
    x = np.arange(1, 14)
    a = -0.5 # realistic range: [-0.535,0]
    b = 5.4 # realistic range: [5.4,6] 
    c = 103.253 # realistic range: [103.253,105]

    r = 15  # r diameter of the microtubules of the triplets
    l = 600 # l Length of the barrel
    s = 15  # s Length between microtubules of the triplet
    nbPts = 1000
    angle = 110 # angle of the triplets

    V = simu_cylinder_rev(get_radius(x, a, b, c), r, l, s, nbPts, angle)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control a, b, c.
    a_slider = Slider(
        ax=plt.axes([0.1, 0.2, 0.8, 0.01]),
        label='a',
        valmin=-0.535,
        valmax=0,
        valinit=a,
    )

    b_slider = Slider(
        ax=plt.axes([0.1, 0.18, 0.8, 0.01]),
        label='b',
        valmin=5.4,
        valmax=6,
        valinit=b,
    )

    c_slider = Slider(
        ax=plt.axes([0.1, 0.16, 0.8, 0.01]),
        label='c',
        valmin=50,
        valmax=200,
        valinit=c,
    )

    r_slider = Slider(
        ax=plt.axes([0.1, 0.14, 0.8, 0.01]),
        label='r',
        valmin=10,
        valmax=20,
        valinit=r,
    )

    l_slider = Slider(
        ax=plt.axes([0.1, 0.12, 0.8, 0.01]),
        label='l',
        valmin=10,
        valmax=1000,
        valinit=l,
    )
    
    s_slider = Slider(
        ax=plt.axes([0.1, 0.10, 0.8, 0.01]),
        label='s',
        valmin=10,
        valmax=20,
        valinit=s,
    )

    angle_slider = Slider(
        ax=plt.axes([0.1, 0.08, 0.8, 0.01]),
        label='angle',
        valmin=0,
        valmax=180,
        valinit=angle,
    )

    X, Y, Z = V[:,0], V[:,1], V[:,2]
    sc = ax.scatter(X, Y, Z, s=0.1)

    t0 = time()
    def update(val):
        global t0
        radius = get_radius(x, a_slider.val, b_slider.val, c_slider.val)
        V = simu_cylinder_rev(radius, r_slider.val, l_slider.val, s_slider.val, nbPts, angle_slider.val)
        sc._offsets3d = (V[:,0], V[:,1], V[:,2])
        t = time()
        print(f"FPS : {1/(t-t0):.2f}")
        t0 = t
        #fig.canvas.draw_idle()

    a_slider.on_changed(update)
    b_slider.on_changed(update)
    c_slider.on_changed(update)
    r_slider.on_changed(update)
    l_slider.on_changed(update)
    s_slider.on_changed(update)
    angle_slider.on_changed(update)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()