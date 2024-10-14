import h5py

def print_structure(h5file, path='/'):
    """递归函数来打印HDF5文件的结构"""
    for key in h5file[path].keys():
        print(f'{path}{key}/')
        if isinstance(h5file[path + key], h5py.Group):  # 检查这个key是否是一个组
            print_structure(h5file, path + key + '/')