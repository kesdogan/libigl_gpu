import torch 

# SNAP_POINTS snap list of points C 
# to closest of another list of points V 
# [I,minD,VI] = snap_points(C,V)

def snap_points(c, v):
    
    # calculate the euclidian distance of each point in c to each point in v
    # minimize and return index and distance
    dist = torch.cdist(c, v)
    min_dist, min_index = torch.min(dist, dim=1)
    
    del dist

    # return additionally list of closest new point positions
    return min_index, min_dist, v[min_index, :]



