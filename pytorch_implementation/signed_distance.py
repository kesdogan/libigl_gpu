### Implemention of https://libigl.github.io/tutorial/#signed-distance

import igl
import torch
from per_face_normals import per_face_normals



# Parameters	vertices #V by 3 tensor of mesh vertex 3D positions
#               faces #F by 3 tensor of face (triangle) indices
#               points #P by 3 tensor of query point positions
#               return_libigl: if True, return the results from libigl
#               extended_vertices: if True, add the center of each edge to the list of vertices (makes for even more accurate results)
#               winding_number: if True, use the winding number to identify the sign of the distance (slower, but most accurate)
def signed_distance(points, vertices, faces, return_libigl=False, extended_vertices=True, winding_number=False):
    
    # in case we want the libigl results, they will be returned here 
    if return_libigl:
        true_S, true_I, true_C = igl.signed_distance(points.cpu().numpy(), vertices.cpu().numpy(), faces.cpu().numpy())
        return true_S, true_I, true_C

    # we can expand the number of vertices by adding the center of each edge to the list of vertices
    # this leads to a better approximation of the closest point on the surface of the mesh
    if extended_vertices:
        vertices =  torch.cat((vertices, 
                    (vertices[faces[:,0]] + vertices[faces[:,2]]) / 2, 
                    (vertices[faces[:,1]] + vertices[faces[:,2]]) / 2), dim=0)

    # in order to identify the closest face to each point, 
    # we compute the distance from each point to the center of each face
    # this works quite well in practices and returns us the face index min_fidx
    average_per_face =  torch.sum(vertices[faces], dim=1)/3
    face_dist = torch.cdist(points, average_per_face)
    _, min_fidx = torch.min(face_dist, dim=1)
    del face_dist

    # identify the sign of the distance to the mesh
    # negative means inside, positive means outside
    # first approach: 
    # winding_number, as detailed in this paper: 
    # https://igl.ethz.ch/projects/winding-number/robust-inside-outside-segmentation-using-generalized-winding-numbers-siggraph-2013-jacobson-et-al.pdf
    # this approach is very slow, but it is the most accurate
    # second approach: 
    # comput the normal of each face, use the matching face indices for each point to identify the normal of the closest face
    # and comput the dot product between the normal and the vector from the point to the center of the face
    # much faster approach, but slightly less accurate
    if winding_number: 
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
        sign = -torch.sign(winding-0.5); del winding, solid_angle
    else: 
        # compute the sign of the distance
        normals = per_face_normals(vertices, faces)
        need_normals = normals[min_fidx]
        product = torch.sum(need_normals * (points - average_per_face[min_fidx]), dim=1)
        sign = torch.sign(product)

    # compute the distance of each point to each vertex (or extended set of vertices)
    dist = torch.cdist(points, vertices)
    min_dist, min_index = torch.min(dist, dim=1)
    del dist

    # default: assign each point to the closest vertex
    p00 = vertices[min_index]

    # implementation of these instructions: 
    # https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    # for each face, we compute whether the closest point on the mesh lies within the triangle 
    # or on one of the edges of the triangle
    # we then identify the closest point there 
    B = vertices[faces[min_fidx, 0]]
    E_0 = vertices[faces[min_fidx, 1]] - vertices[faces[min_fidx, 0]]
    E_1 = vertices[faces[min_fidx, 2]] - vertices[faces[min_fidx, 0]]
    a = torch.sum(E_0 * E_0, dim=1); b = torch.sum(E_0 * E_1, dim=1);    c = torch.sum(E_1 * E_1, dim=1)
    d = torch.sum(E_0 * (vertices[faces[min_fidx, 0]] - points), dim=1); e = torch.sum(E_1 * (vertices[faces[min_fidx, 0]] - points), dim=1)
    det = a * c - b * b; s = (b * e - c * d); t = (b * d - a * e)

    reg0 = (s > 0) * (t > 0) * (s + t < det) # inside the triangle
    reg1 = (s > 0) * (t > 0) * (s + t > det) # outside of the top edge 
    reg3 = (s < 0) * (t > 0) * (s + t < det) # outside of the left edge 
    reg5 = (s > 0) * (t < 0) * (s + t < det) # outside of the bottom edge 
    
    # region0
    s[reg0] = s[reg0] / det[reg0]; t[reg0] = t[reg0] / det[reg0]
    
    # region1
    numer = (c + e) - (b + d);          denom = a - 2 * b + c
    check_numer = numer <= 0;           o_check_numer = numer > 0
    check_denom_numer = numer >= denom; o_check_denom_numer = numer < denom
    s[reg1*check_numer] = 0;            s[reg1*check_denom_numer*o_check_numer] = 1
    s[reg1*o_check_denom_numer*o_check_numer] = (numer[reg1*o_check_denom_numer*o_check_numer] / 
                                                denom[reg1*o_check_denom_numer*o_check_numer])
    t[reg1] = 1 - s[reg1]
    
    # region3
    check_e = e >= 0;       o_check_e = e < 0
    check_e_c = -e >= c;    o_check_e_c = -e < c
    s[reg3] = 0;            t[reg3*check_e] = 0
    t[reg3*o_check_e*check_e_c] = 1
    t[reg3*o_check_e*o_check_e_c] = (-e[reg3*o_check_e*o_check_e_c] / 
                                    c[reg3*o_check_e*o_check_e_c])
    
    #region5
    check_d = d >= 0;       o_check_d = d < 0
    check_d_a = -d >= a;    o_check_d_a = -d < a
    t[reg5] = 0;            s[reg5*check_d] = 0
    s[reg5*o_check_d*check_d_a] = 1
    s[reg5*o_check_d*o_check_d_a] = (-d[reg5*o_check_d*o_check_d_a] / 
                                    a[reg5*o_check_d*o_check_d_a])
    
    # correct closest point for those lying on the edges and inside the triangle
    p00[reg0+reg1+reg3+reg5] = (B[reg0+reg1+reg3+reg5] + s[reg0+reg1+reg3+reg5, None] * E_0[reg0+reg1+reg3+reg5] + 
                                t[reg0+reg1+reg3+reg5, None] * E_1[reg0+reg1+reg3+reg5])
    min_dist[reg0+reg1+reg3+reg5] = torch.norm(p00[reg0+reg1+reg3+reg5] - points[reg0+reg1+reg3+reg5], dim=1)

    return sign*min_dist, min_fidx, p00
