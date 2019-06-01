# import h5py
# filename = 'tempFiles/temp.hdf5'
# f = h5py.File(filename, 'r')

# for key in f.keys():
#     print(key) #Names of the groups in HDF5 file.

import h5py
filename = 'tempFiles/temp.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])