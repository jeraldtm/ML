'''
Created on 28 Feb 2018

@author: jerald
'''
import h5py
filename = 'my_model.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
print(data)