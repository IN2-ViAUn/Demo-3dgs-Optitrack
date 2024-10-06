import math

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def simple_render(viewpoint_camera, pc, bg_color: torch.tensor):
    '''
    Simplest gaussian rendering 
    '''
    #  screenspace_points = torch.zeros_like(pc.mean3D, dtype=pc.mean3D.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros_like(pc.mean3D, dtype=pc.mean3D.dtype, requires_grad=False, device="cuda") + 0
    # why add 0 ?
    #  screenspace_points = torch.zeros_like(pc.mean3D, dtype=pc.mean3D.dtype, requires_grad=True, device="cuda")

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,

        #  scale_modifier=scaling_modifier,
        scale_modifier=1,
        sh_degree=pc.active_sh_degree,
        prefiltered=False,
        #  debug=pipe.debug
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    #  breakpoint()

    means3D = pc.mean3D
    means2D = screenspace_points
    opacity = pc.opacities
    scales = pc.scales
    rotations = pc.rots
    shs = pc.features

    rendered_image, _radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        #  colors_precomp = colors_precomp,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        #  cov3D_precomp = cov3D_precomp)
        cov3D_precomp = None)

    #  _ = _radii>0

    return rendered_image
