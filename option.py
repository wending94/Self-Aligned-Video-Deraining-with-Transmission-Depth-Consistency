
class item:
    def __init__(self):
        self.name = ''

opt = item()

opt.checkpoint_dir = './checkpoints/'
opt.data_dir = '/home/disk_wending/Data/Rain_Data'
opt.list_filename = './lists/video_rain_removal_train.txt'
opt.test_list_filename = './lists/video_rain_removal_test.txt'
opt.self_tag = 'P401_video_rain_self'

# opt.model_name = 'derain_self_v4_flow_rb1_v2'
opt.batch_size = 1
opt.crop_size = 64
opt.vgg_path = '/home/yanwending/.cache/torch/checkpoints/vgg16-397923af.pth'

opt.threads = 8
opt.input_show = False

opt.train_epoch_size = 500
opt.valid_epoch_size = 100
opt.epoch_max = 100
