from numpy import concatenate, array, lexsort, zeros, log2
from joblib import Parallel, delayed

__all__ = [
    'int_numpy', 'float_numpy', 'dim_space', 'sorted_cloud', 'mipmap',
    'mip_grid', 'voxel_grid', 'mip_cloud']

int_numpy = 'int64'
float_numpy = 'float32'
dim_space = 3

class sorted_cloud(object):
    
    def __init__(self, cloud=None, feature=None, space=None, dtype=None):
        if dtype is None:
            dtype = float_numpy
        if space is None:
            space = dim_space
        if feature is None and cloud is None:
            feature = 0
            cloud = zeros((0, space), dtype)
        elif feature is None:
            feature = cloud.shape[1] - space
        elif cloud is None:
            cloud = zeros((0, space + feature), dtype)
        self.dtype = dtype
        self.space = space
        self.feature = feature
        self.clear()
        self.extend(cloud)
    
    def clear(self):
        self.data = zeros((0, self.space * 2 + self.feature), self.dtype)
        self.bound = zeros((1,), int_numpy)
    
    def extend(self, cloud):
        self.data = concatenate((
            self.data,
            concatenate((cloud[:, :self.space] // 1., cloud), 1),
        ))
        self.data = self.data[lexsort([
            self.data[:, i] for i in range(self.space)])]
        bound = [0]
        for i in range(1, len(self.data)):
            if any(self.data[i, :self.space] != self.data[i-1, :self.space]):
                bound.append(i)
        bound.append(len(self.data))
        self.bound = array(bound, int_numpy)
    
    def __getitem__(self, key):
        if 0 > key:
            key -= 1
        return (
            array(self.data[self.bound[key], :self.space]),
            array(self.data[self.bound[key]:self.bound[key+1], self.space:]),
        )
    
    def __len__(self):
        return len(self.bound) - 1

class mipmap(object):
    
    def __init__(self, index, cloud, space, fineness=None, dtype=None):
        if dtype is None:
            dtype = float_numpy
        if fineness is None:
            fineness = 11
        self.dtype = dtype
        self.fineness = fineness
        self.feature = cloud.shape[1] - space
        self.space = space
        self.bound = zeros((1, space + fineness + 1), int_numpy)
        self.data = zeros((0, cloud.shape[1]), dtype)
        self.count = zeros(fineness, int_numpy)
        self.bound[0, :space] = index
        for level in range(fineness):
            cloud_vox = voxel_grid(cloud, 0.5 ** level, space)
            self.data = concatenate((self.data, cloud_vox[:, :-1]))
            self.count[level] += len(cloud_vox)
            self.bound[0, space+level+1] = (
                self.bound[0, space+level] + len(cloud_vox))
    
    def __getitem__(self, key):
        if type(key) == int:
            key = (key,)
        if len(key) == 1:
            return array(self.bound[key[0], :self.space])
        elif len(key) > 1:
            return array(self.data[
                self.bound[key[0], self.space+key[1]]:
                self.bound[key[0], self.space+key[1]+1]])
    
    def __len__(self):
        return len(self.bound)
    
    def merge(self, value):
        self.count += value.count
        self.data = concatenate((self.data, value.data))
        if len(self.bound):
            value.bound[self.space:] += self.bound[-1, -1]
        self.bound = concatenate((self.bound, value.bound))

class mip_grid(mipmap):
    
    def __init__(self, source, n_jobs=1, fineness=None, dtype=None):
        if dtype is None:
            dtype = float_numpy
        if fineness is None:
            fineness = 11
        self.dtype = dtype
        self.fineness = fineness
        self.feature = source.feature
        self.space = source.space
        self.bound = zeros((0, self.space + fineness + 1), int_numpy)
        self.data = zeros((0, self.space + self.feature), dtype)
        self.count = zeros(fineness, int_numpy)
        for mip in Parallel(n_jobs)(delayed(self.temp_mip)(
            *source[i], self.space, fineness
        ) for i in range(len(source))):
            self.merge(mip)
    
    def temp_mip(self, *args, **kwargs):
        mip = mipmap(*args, **kwargs)
        return mip

def voxel_grid(cloud, leaf, space=None, dtype=None):
    if dtype is None:
        dtype = float_numpy
    if space is None:
        space = dim_space
    grid = {}
    for pt in cloud:
        vox = tuple(pt[:space] // leaf)
        if vox in grid:
            grid[vox] += concatenate((pt, (1.0,)))
        else:
            grid[vox] = concatenate((pt, (1.0,)))
    cloud_vox = array(tuple(grid.values()), dtype)
    cloud_vox[:, :-1] /= cloud_vox[:, -1:]
    return cloud_vox

def mip_cloud(cloud, feature=None, space=None, n_jobs=1, fineness=None):
    return mip_grid(sorted_cloud(cloud, feature, space), n_jobs, fineness)

def cloud_mip(mip, center, f):
    cloud = zeros((0, mip.space + mip.feature))
    for index in mip:
        level = min(max(
            -int(log2((index + array((0.5, 0.5, 0.5)) - center) / f) // 1.),
            0), mip.fineness - 1)
        cloud.concatenate(cloud, mip[index, level])
    return cloud
