# The hyper-parameter configerations of Segmenter-SAPS

# Data
classes = 17
dataset = 2

# Model
model = "deit_small_distilled_patch16_224"
decoder = "mask_transformer"
div_thres = 5 * 255
min_patchsize = 4

# Train
momentum = 0.9
weight_decay = 0.0001
drop_out_rate = 0.0
drop_path_rate = 0.1
