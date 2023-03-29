import numpy as np


# data: 3-D array, BGR image
# buf: 3-D array, H*W*one-hot code
def bgr2onehot(data):
    bgrdic = {(0,0,0):0, (0,0,255):1, (194,0,243):2, (0,243,194):3, (165,113,243):4, (0,188,0):5, 
              (97,243,0):6, (0,75,188):7, (255,128,96):8, (243,0,0):9, (15,159,255):10, (243,113,13):11, 
              (100,50,100):12, (143,235,239):13, (223,255,96):14, (175,191,64):15, (215,0,43):16}
    n = len(bgrdic)
    buf = np.zeros(data.shape[0:2] + (n, ))
    (rows, cols) = data.shape[0:2]
    
    for i in range(rows):
        for j in range(cols):
            dicidx = tuple(data[i,j])
            buf[i, j, bgrdic[dicidx]] = 1
    
    return buf

# The following 2 functions are modified from bgr2onehot() and onehot2mbatchrgb() since the loss
# function of CrossEntropyLoss() doesn't support one-hot code.

# data: 3-D array, BGR image
# buf: 2-D array, H*W where each value is >= 0 and <= len(bgrdic)-1
def bgr2class(data):
    bgrdic = {(0,0,0):0, (0,0,255):1, (194,0,243):2, (0,243,194):3, (165,113,243):4, (0,188,0):5, 
              (97,243,0):6, (0,75,188):7, (255,128,96):8, (243,0,0):9, (15,159,255):10, (243,113,13):11, 
              (100,50,100):12, (143,235,239):13, (223,255,96):14, (175,191,64):15, (215,0,43):16}
    buf = np.zeros(data.shape[0:2])
    (rows, cols) = data.shape[0:2]
    
    for i in range(rows):
        for j in range(cols):
            dicidx = tuple(data[i,j])
            buf[i, j] = bgrdic[dicidx]
    
    return buf

# data: 4-D array, N*onehot*H*W
# output: 4-D array, N*3*H*W
def onehot2mbatchrgb(data):
    rgbdic = {0:[0,0,0], 1:[255,0,0], 2:[243,0,194], 3:[194,243,0], 4:[243,113,165], 5:[0,188,0], 
              6:[0,243,97], 7:[188,75,0], 8:[96,128,255], 9:[0,0,243], 10:[255,159,15], 11:[13,113,243], 
              12:[100,50,100], 13:[239,235,143], 14:[96,255,223], 15:[64,191,175], 16:[43,0,215]}
    data = np.argmax(data, axis=1)
    output = np.zeros(data.shape[0:3] + (3, ))
    (batches, rows, cols) = data.shape[0:3]
    
    for b in range(batches):
        for i in range(rows):
            for j in range(cols):
                output[b,i,j] = rgbdic[data[b,i,j]]
    output = output.transpose(0,3,1,2)
                
    return output

# data: 3-D array, N*H*W
# output: 4-D array, N*3*H*W
def class2mbatchrgb(data):
    rgbdic = {0:[0,0,0], 1:[255,0,0], 2:[243,0,194], 3:[194,243,0], 4:[243,113,165], 5:[0,188,0], 
              6:[0,243,97], 7:[188,75,0], 8:[96,128,255], 9:[0,0,243], 10:[255,159,15], 11:[13,113,243], 
              12:[100,50,100], 13:[239,235,143], 14:[96,255,223], 15:[64,191,175], 16:[43,0,215]}
    output = np.zeros(data.shape[0:3] + (3, ))
    (batches, rows, cols) = data.shape[0:3]
    
    for b in range(batches):
        for i in range(rows):
            for j in range(cols):
                output[b,i,j] = rgbdic[data[b,i,j]]
    output = output.transpose(0,3,1,2)
                
    return output

# data: 4-D array, N*onehot*H*W
# output: 3-D array, N*H*W
def onehot2mbatchclass(data):
    output = np.argmax(data, axis=1)
    return output
