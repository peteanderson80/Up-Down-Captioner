#!/usr/bin/env python
"""
Utils
"""

import h5py


def restore_weights(hdf5_path):
    ''' Copy weights back from beam search to original layer names '''    
    mapping = {
      'beam' : [
        ('embedding', [0]),
        ('lstm1', [0,1]),
        ('hidden_att_0', [0]),
        ('predict_att_0', [0]),
        ('lstm2', [0,1]),
        ('predict', [0,1])
      ]
    }
    f = h5py.File(hdf5_path,'a')
    for src,dest_list in mapping.iteritems():
        src_index = 0
        for dest,blobs in dest_list:
            for blob_index in blobs:
                from_path = '/data/'+src+'/'+str(src_index)
                to_path = '/data/'+dest+'/'+str(blob_index)
                print 'Attempting copy: %s -> %s' % (from_path,to_path)
                try:
                  weights = f[from_path]
                  f.create_dataset(to_path, data=weights)
                except Exception as e:
                  print '...failed: %s' % e
                src_index += 1
    f.close()
