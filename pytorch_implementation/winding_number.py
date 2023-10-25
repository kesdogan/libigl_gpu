### Implemention of https://libigl.github.io/tutorial/#winding_number

### For further reading, see the following paper:
### https://igl.ethz.ch/projects/winding-number/robust-inside-outside-segmentation-using-generalized-winding-numbers-siggraph-2013-jacobson-et-al.pdf

import torch


# Parameters	vertices #V by 3 tensor of mesh vertex 3D positions
#               faces #F by 3 tensor of face (triangle) indices
#               points #P by 3 tensor of query point positions

# Returns	    winding #P by 1 tensor of winding numbers
def winding_number(vertices, faces, points):
    triangles = vertices[faces].unsqueeze(0).repeat(points.shape[0], 1, 1, 1)
    abc = triangles - points.unsqueeze(1).unsqueeze(1)
    del triangles
    norm = torch.norm(abc, dim=3)
    solid_angle = torch.atan2(  abc[:, :, 0, 0] * abc[:, :, 1, 1] * abc[:, :, 2, 2] +
                                abc[:, :, 0, 1] * abc[:, :, 1, 2] * abc[:, :, 2, 0] +
                                abc[:, :, 0, 2] * abc[:, :, 1, 0] * abc[:, :, 2, 1] -
                                abc[:, :, 0, 2] * abc[:, :, 1, 1] * abc[:, :, 2, 0] -
                                abc[:, :, 0, 1] * abc[:, :, 1, 0] * abc[:, :, 2, 2] -
                                abc[:, :, 0, 0] * abc[:, :, 1, 2] * abc[:, :, 2, 1],
                            (torch.sum(abc[:,:,0,:] * abc[:,:,1,:], dim = 2) * norm[:,:,2] +
                            torch.sum(abc[:,:,0,:] * abc[:,:,2,:], dim = 2) * norm[:,:,1] +
                            torch.sum(abc[:,:,1,:] * abc[:,:,2,:], dim = 2) * norm[:,:,0]))
    del abc, norm
    winding = torch.sum(solid_angle, dim=1) / (2 * torch.pi)

    return winding