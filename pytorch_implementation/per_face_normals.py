### implementation of https://libigl.github.io/libigl-python-bindings/igl_docs/#per_face_normals

import torch


# Parameters	V #V by 3 Matrix of mesh vertex 3D positions
#               F #F by 3 Matrix of face (triangle) indices

# Returns	    N #F by 3 Matrix of mesh face (triangle) 3D normals

def per_face_normals(vertices, faces):
    face_normals = torch.cross(vertices[faces[:,1]] - vertices[faces[:,0]], vertices[faces[:,2]] - vertices[faces[:,0]])
    face_normals = face_normals / torch.norm(face_normals, dim=1).unsqueeze(1)
    return face_normals