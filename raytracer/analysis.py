"""Analysis module."""
import matplotlib.pyplot as plt
from raytracer.elements import SphericalRefraction, OutputPlane, SphericalReflection
from raytracer.rays import Ray, RayBundle
from raytracer.lenses import BiConvex, LensMaker
import numpy as np
from scipy.optimize import minimize 

default_refractor = {
    "z_0": 100,
    "aperture": 34,
    "curvature": 0.03,
    "n_1": 1,
    "n_2": 1.5
}


def task8():
    """
    Task 8.

    In this function you should check your propagate_ray function properly
    finds the correct intercept and correctly refracts a ray. Don't forget
    to check that the correct values are appended to your Ray object.
    """
    test_ray = Ray(direc=[0, 1, 10])
    test_ray1 = Ray(direc=[1, 0, 10])

    refractor = SphericalRefraction(**default_refractor)
    refractor.propagate_ray(test_ray)
    refractor.propagate_ray(test_ray1)
    
    vertices = test_ray.vertices()
    vertices1 = test_ray1.vertices()
    all_vertices = vertices + vertices1
    points = np.vstack(all_vertices)
    
    x = points[:,0]
    y = points[:, 1]
    z = points[:, 2]    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    refractor.plot_surface(ax = ax)
    ax.plot(x, y, z, marker='o') 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    ax.set_title('Ray Vertices Path task 8')

    plt.show()
    

def task10():
    """
    Task 10.

    In this function you should create Ray objects with the given initial positions.
    These rays should be propagated through the surface, up to the output plane.
    You should then plot the tracks of these rays.
    This function should return the matplotlib figure of the ray paths.

    Returns:
        Figure: the ray path plot.
    """
    
    positions = [
    [0, 4, 0],
    [0, 1, 0],
    [0, 0.2, 0],
    [0, 0, 0],
    [0, -0.2, 0],
    [0, -1, 0],
    [0, -4, 0]
    ]
    direc = [0, 0, 1]
    rays = [Ray(pos=pos, direc=direc) for pos in positions]
    refractor = SphericalRefraction(**default_refractor)
    output = OutputPlane(z_0 = 250)
    
    [refractor.propagate_ray(ray) for ray in rays]
    [output.propagate_ray(ray) for ray in rays]
    vertices = [v for ray in rays for v in ray.vertices()]
    points = np.vstack(vertices)

    
    x = points[:,0]
    y = points[:, 1]
    z = points[:, 2]    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    refractor.plot_surface(ax = ax)
    ax.plot(x, y, z, marker='o') 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    ax.set_title('Ray Vertices Path task 10')
    
    return fig


def task11():
    """
    Task 11.

    In this function you should propagate the three given paraxial rays through the system
    to the output plane and the tracks of these rays should then be plotted.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for ray paths
    2. the calculated focal point.

    Returns:
        tuple[Figure, float]: the ray path plot and the focal point
    """
    positions = [
    [0.1, 0.1, 0],
    [0, 0, 0],
    [-0.1, -0.1, 0]
    ]
    
    direc = [0, 0, 1]
    rays = [Ray(pos=pos, direc=direc) for pos in positions]
    refractor = SphericalRefraction(z_0 = 100, aperture = 34, curvature = 0.03, n_1 = 1, n_2 = 1.5)
    focal_point = refractor.focal_point()
    output = OutputPlane(z_0 = focal_point)
    
    [refractor.propagate_ray(ray) for ray in rays]
    [output.propagate_ray(ray) for ray in rays]
    vertices = [v for ray in rays for v in ray.vertices()]
    points = np.vstack(vertices)

    
    x = points[:,0]
    y = points[:, 1]
    z = points[:, 2]    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    refractor.plot_surface(ax = ax)
    ax.plot(x, y, z, marker='o') 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    ax.set_title('Ray Vertices Path task 11')
    
    return [fig, focal_point]


def task12():
    """
    Task 12.

    In this function you should create a RayBunble and propagate it to the output plane
    before plotting the tracks of the rays.
    This function should return the matplotlib figure of the track plot.

    Returns:
        Figure: the track plot.
    """
    bundle = RayBundle()
    refractor = SphericalRefraction(**default_refractor)
    focal_point = refractor.focal_point()
    output = OutputPlane(z_0 = focal_point)
    
    bundle.propagate_bundle([refractor, output])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    bundle.track_plot(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore

    ax.set_title('Ray Vertices Path task 12')  

    refractor.plot_surface(ax)
    return fig


def task13():
    """
    Task 13.

    In this function you should again create and propagate a RayBundle to the output plane
    before plotting the spot plot.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot
    2. the simulation RMS

    Returns:
        tuple[Figure, float]: the spot plot and rms
    """
    bundle = RayBundle()
    refractor = SphericalRefraction(**default_refractor)
    focal_point = refractor.focal_point()
    output = OutputPlane(z_0 = focal_point)
    
    bundle.propagate_bundle([refractor, output])
    fig = bundle.spot_plot()
    fig.axes[0].set_title("task 13 spot plot")
    rms = bundle.rms()
    
    return [fig, rms]


def task14():
    """
    Task 14.

    In this function you will trace a number of RayBundles through the optical system and
    plot the RMS and diffraction scale dependence on input beam radii.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the simulation RMS for input beam radius 2.5
    3. the diffraction scale for input beam radius 2.5

    Returns:
        tuple[Figure, float, float]: the plot, the simulation RMS value, the diffraction scale.
    """
    wavelength = 588e-6  # mm

    radius_arr = np.arange(0.1, 10.0, 0.1)

    refractor = SphericalRefraction(**default_refractor)
    focal_point = refractor.focal_point()
    output = OutputPlane(z_0 = focal_point)
    focal_length = focal_point - refractor.z_0()

    rms_arr = []
    for r in radius_arr:
        bundle = RayBundle(rmax = r)
        bundle.propagate_bundle([refractor, output])
        rms_arr.append(bundle.rms())

    rms_arr = np.array(rms_arr)
    diff_scale_arr = wavelength * focal_length / (2 * radius_arr)
    mask = np.isclose(radius_arr, 2.5)
    i = np.where(mask)[0]
    
    standard_rms = rms_arr[i][0]
    standard_diff = diff_scale_arr[i][0]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(radius_arr, rms_arr, label = 'RMS Spread', color = 'blue')
    ax.plot(radius_arr, diff_scale_arr, label = '∆x Scale', color = 'red')
    ax.set_xlabel('Beam Radius')
    ax.set_title('RMS vs Diffraction Scale task 14')
    ax.legend()

    return [fig, standard_rms, standard_diff]


def task15():
    """
    Task 15.

    In this function you will create plano-convex lenses in each orientation and propagate a RayBundle
    through each to their respective focal point. You should then plot the spot plot for each orientation.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot for the plano-convex system
    2. the focal point for the plano-convex lens
    3. the matplotlib figure object for the spot plot for the convex-plano system
    4  the focal point for the convex-plano lens


    Returns:
        tuple[Figure, float, Figure, float]: the spot plots and rms for plano-convex and convex-plano.
    """
    
    pc = LensMaker(z_0=100, curvature1=0., curvature2=-0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    focal_length_pc = pc.focal_point() + pc.z_0() + pc.thickness()
    output_pc = OutputPlane(z_0=focal_length_pc)
    
    bundle_pc = RayBundle() 
    bundle_pc.propagate_bundle([pc, output_pc])
    fig_pc = bundle_pc.spot_plot()
    fig_pc.axes[0].set_title('Task 15 pc spotplot')
    
    
    cp = LensMaker(z_0=100, curvature1=0.02, curvature2=0., n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    focal_length_cp = cp.focal_point() + cp.z_0() + cp.thickness()
    output_cp = OutputPlane(z_0=focal_length_cp)
    
    bundle_cp = RayBundle()
    bundle_cp.propagate_bundle([cp, output_cp])
    fig_cp = bundle_cp.spot_plot()
    fig_cp.axes[0].set_title('Task 15 cp spotplot')
    
    return [fig_pc, pc.focal_point(), fig_cp, cp.focal_point()]


def task16():
    """
    Task 16.

    In this function you will be again plotting the radial dependence of the RMS and diffraction values
    for each orientation of your lens.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the RMS for input beam radius 3.5 for the plano-convex system
    3. the RMS for input beam radius 3.5 for the convex-plano system
    4  the diffraction scale for input beam radius 3.5

    Returns:
        tuple[Figure, float, float, float]: the plot, RMS for plano-convex, RMS for convex-plano, diffraction scale.
    """
    wavelength = 588e-6  # mm

    radius_arr = np.arange(0.1, 10.0, 0.1)

    pc = LensMaker(z_0=100., curvature1=0., curvature2=-0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    focal_point_pc = pc.focal_point()
    output_pc = OutputPlane(z_0 = focal_point_pc)
    focal_length_pc = pc.focal_length()

    rms_arr_pc = []
    for r in radius_arr:
        bundle_pc = RayBundle(rmax = r)
        bundle_pc.propagate_bundle([pc, output_pc])
        rms_arr_pc.append(bundle_pc.rms())

    rms_arr_pc = np.array(rms_arr_pc)
    diff_scale_arr_pc = wavelength * focal_length_pc / (2 * radius_arr)
    mask = np.isclose(radius_arr, 3.5)
    i = np.where(mask)[0]
    
    standard_diff_pc = diff_scale_arr_pc[i][0]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.plot(radius_arr, rms_arr_pc, label = 'RMS Spread', color = 'blue')
    ax.plot(radius_arr, diff_scale_arr_pc, label = '∆x Scale', color = 'red')
    ax.set_xlabel('Beam Radius')
    ax.set_title('RMS vs Diffraction Scale (Task 16)')
    ax.legend()
    
    cp = LensMaker(z_0=100., curvature1=0.02, curvature2=0., n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    focal_point_cp = cp.focal_point()
    output_cp = OutputPlane(z_0 = focal_point_cp)
    focal_length_cp = cp.focal_length()

    rms_arr_cp = []
    for r in radius_arr:
        bundle_cp = RayBundle(rmax = r)
        bundle_cp.propagate_bundle([cp, output_cp])
        rms_arr_cp.append(bundle_cp.rms())

    rms_arr_cp = np.array(rms_arr_cp)
    diff_scale_arr_cp = wavelength * focal_length_cp / (2 * radius_arr)
    
    diff35 = wavelength * focal_length_cp / (2 * 3.5)
    
    ax.plot(radius_arr, rms_arr_cp, label = 'RMS Spread', color = 'blue')
    ax.plot(radius_arr, diff_scale_arr_cp, label = '∆x Scale', color = 'red')
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d') 
    bundle_pc.track_plot(ax1) 
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')  # type: ignore  
    pc.plot_lens(ax = ax1)
    ax1.set_title("task 16 pc trackplot")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d') 
    bundle_pc.track_plot(ax2) 
    print(type(cp), type(pc))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')  # type: ignore  
    cp.plot_lens(ax = ax2)
    ax2.set_title("task 16 cp trackplot")
    

    return [fig, rms_arr_pc[i][0], rms_arr_cp[i][0], diff35]


def task17():
    """
    Task 17.

    In this function you will be first plotting the spot plot for your PlanoConvex lens with the curved
    side first (at the focal point). You will then be optimising the curvatures of a BiConvex lens
    in order to minimise the RMS spot size at the same focal point. This function should return
    the following items as a tuple in the following order:
    1. The comparison spot plot for both PlanoConvex (curved side first) and BiConvex lenses at PlanoConvex focal point.
    2. The RMS spot size for the PlanoConvex lens at focal point
    3. the RMS spot size for the BiConvex lens at PlanoConvex focal point

    Returns:
        tuple[Figure, float, float]: The combined spot plot, RMS for the PC lens, RMS for the BiConvex lens
    """
    bundle_cp = RayBundle()
    
    cpl = LensMaker(z_0=100., curvature1=0.02, curvature2=0., n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    cpl_foc = cpl.focal_point()
    output_plane = OutputPlane(cpl_foc)
    bundle_cp.propagate_bundle([cpl, output_plane])
    
    def objective(curvatures):
        c1, c2 = curvatures
        lens = BiConvex(z_0=100., curvature1=c1, curvature2=c2, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
        bundle = RayBundle()
        bundle.propagate_bundle([lens, output_plane])
        return bundle.rms()
    
    res = minimize(objective, x0=[0.0146, -0.00563], options={'maxiter': 100000000000000000000, 'disp': False})
    biconvex_lens = BiConvex(z_0=100., curvature1=res.x[0], curvature2=res.x[1], n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)

    bundle_bi = RayBundle(rmax=3.5)
    bundle_bi.propagate_bundle([biconvex_lens, output_plane])
    
    fig = plt.figure()
    ax = fig.add_subplot()

    bundle_bi.spot_plot(ax)
    bundle_cp.spot_plot(ax)
    plt.xlabel('x position (mm)')
    plt.ylabel('y position (mm)')
    plt.title('Spot Plot')
    plt.grid(True)
    
    return [fig, bundle_cp.rms(), bundle_bi.rms()]

def task18():
    """
    Task 18.

    In this function you will be testing your reflection modelling. Create a new SphericalReflecting surface
    and trace a RayBundle through it to the OutputPlane.This function should return
    the following items as a tuple in the following order:
    1. The track plot showing reflecting ray bundle off SphericalReflection surface.
    2. The focal point of the SphericalReflection surface.

    Returns:
        tuple[Figure, float]: The track plot and the focal point.

    """
    sph_refl = SphericalReflection(z_0=50., aperture=30., curvature=-0.02)
    output = OutputPlane(z_0=30)
    bundle = RayBundle(rmax=5)
    bundle.propagate_bundle([sph_refl, output])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore  
    
    sph_refl.plot_surface(ax)
    bundle.track_plot(ax)
    ax.set_title('Ray Vertices Path task 18')
    

    return [fig, sph_refl.focal_point()]


if __name__ == "__main__":

    # Run task 8 function
    # task8()

    # # Run task 10 function
    # FIG10 = task10()

    # # Run task 11 function
    # FIG11, FOCAL_POINT = task11()

    # # Run task 12 function
    # FIG12 = task12()

    # # Run task 13 function
    # FIG13, TASK13_RMS = task13()

    # # Run task 14 function
    # FIG14, TASK14_RMS, TASK14_DIFF_SCALE = task14()

    # # Run task 15 function
    # FIG15_PC, FOCAL_POINT_PC, FIG15_CP, FOCAL_POINT_CP = task15()

    # Run task 16 function
    FIG16, PC_RMS, CP_RMS, TASK16_DIFF_SCALE = task16()

    # # Run task 17 function
    # FIG17, CP_RMS, BICONVEX_RMS = task17()

    # # Run task 18 function
    # FIG18, FOCAL_POINT = task18()

    plt.show()
