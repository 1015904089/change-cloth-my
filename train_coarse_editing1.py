import torch
import argparse
from omegaconf import OmegaConf
from models.provider import SphericalSamplingGSDataset
# from models.GPS.Trainer import Trainer as GPSNetwork
from models.trainer_coarse_editing import Trainer_SDS
from models.utils.sh_utils import seed_everything
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_global', default=None, help="global text prompt")
    parser.add_argument('--text_local', default=None, help="local text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_video', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=20, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--bbox_path', type=str, default=None)
    parser.add_argument('--sd_path', type=str, default='./res_gaussion/colmap_doll/content_personalization')
    parser.add_argument('--sd_img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sd_max_step_start', type=float, default=0.75, help="sd_max_step")
    parser.add_argument('--sd_max_step_end', type=float, default=0.25, help="sd_max_step")

    ### training options
    parser.add_argument('--sh_degree', type=int, default=0)
    parser.add_argument('--bbox_size_factor', type=float, default=1., help="size factor of 3d bounding box")
    parser.add_argument('--start_gamma', type=float, default=0.99, help="initial gamma value")
    parser.add_argument('--end_gamma', type=float, default=0.5, help="end gamma value")
    parser.add_argument('--points_times', type=int, default=1, help="repeat editing points x times")
    parser.add_argument('--position_lr_init', type=float, default=0.00015, help="initial learning rate")
    parser.add_argument('--position_lr_final', type=float, default=0.000002, help="initial learning rate")
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--position_lr_max_steps', type=float, default=5_000, help="initial learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.05, help="initial learning rate")
    parser.add_argument('--feature_lr_after', type=float, default=0.02, help="initial learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.05, help="initial learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--percent_dense', type=float, default=0.1, help="0.01")
    parser.add_argument('--densification_interval', type=float, default=200)
    parser.add_argument('--opacity_reset_interval', type=float, default=2000)
    parser.add_argument('--densify_from_iter', type=float, default=500)
    parser.add_argument('--densify_until_iter', type=float, default=800)
    parser.add_argument('--densify_grad_threshold', type=float, default=15, help="0.0002")
    parser.add_argument('--min_opacity', type=float, default=0.005)
    parser.add_argument('--max_screen_size', type=float, default=20.0)
    parser.add_argument('--max_scale_size', type=float, default=0.05)
    parser.add_argument('--extent', type=float, default=4.4)
    parser.add_argument('--iters', type=int, default=3000)
    # parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--sample_num', type=int, default=200, help="GUI width")


    parser.add_argument("--stage1_ckpt", type=str, default=None, help="0:add new object, 1:edit existing object")
    parser.add_argument("--restore_ckpt", type=str, default="", help="0:add new object, 1:edit existing object")
    parser.add_argument("--lr", type=float, default=1e-2, help="0:add new object, 1:edit existing object")
    parser.add_argument("--wdecay", type=float, default=1e-5, help="0:add new object, 1:edit existing object")
    parser.add_argument("--mixed_precision", action='store_true')
    parser.add_argument("--train_iters", type=int, default=3, help="0:add new object, 1:edit existing object")
    parser.add_argument("--val_iters", type=int, default=3, help="0:add new object, 1:edit existing object")
    parser.add_argument('--train_resolution_level', type=float, default=1, help="GUI width")

    parser.add_argument('--raft_encoder_dims', type=float, nargs='*', default=[32, 48, 96], help="training camera fovy range")
    parser.add_argument('--raft_hidden_dims', type=float, nargs='*', default=[96, 96, 96], help="training camera fovy range")
    parser.add_argument('--corr_implementation', type=str, nargs='*', default='reg', help="training camera fovy range")
    parser.add_argument('--corr_levels', type=int, nargs='*', default=4, help="training camera fovy range")
    parser.add_argument('--corr_radius', type=int, nargs='*', default=4, help="training camera fovy range")
    parser.add_argument('--n_gru_layers', type=int, nargs='*', default=1, help="training camera fovy range")
    parser.add_argument('--n_downsample', type=int, nargs='*', default=3, help="training camera fovy range")
    parser.add_argument("--data_type", type=str, default='colmap', help='input data')
    parser.add_argument("--cloth", type=str, default='00006', help='input data')
    parser.add_argument('--gsnet_encoder_dims', type=float, nargs='*', default=[32, 48, 96], help="training camera fovy range")
    parser.add_argument('--gsnet_decoder_dims', type=float, nargs='*', default=[48, 64, 96], help="training camera fovy range")
    parser.add_argument("--parm_head_dim", type=int, default=32, help="0:add new object, 1:edit existing object")


    parser.add_argument("--loss_freq", type=int, default=5000, help="0:add new object, 1:edit existing object")
    parser.add_argument("--eval_freq", type=int, default=5000, help="0:add new object, 1:edit existing object")

    parser.add_argument("--editing_type", type=int, default=1, help="0:add new object, 1:edit existing object")
    parser.add_argument("--reset_points", type=bool, default=False, help="If reset color and size of the editing points")
    parser.add_argument("--only_gs", action='store_true', help="If reset color and size of the editing points")
    parser.add_argument("--fix", action='store_true', help="If reset color and size of the editing points")
    parser.add_argument("--sd", action='store_true', help="If reset color and size of the editing points")


    ### dataset options
    parser.add_argument("--pose_sample_strategy", type=str, default='uniform',
                        help='input data directory')
    parser.add_argument("--R_path", type=str, default=None,
                        help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--num_work', type=int, default=4, help="GUI width")

    parser.add_argument('--data_path', type=str, default="", help="0:add new object, 1:edit existing object")
    parser.add_argument('--initial_points', type=str, default="", help="0:add new object, 1:edit existing object")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[25, 25], help="test camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[-4.5, -4.5], help="test camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="test camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[90, 90], help="test camera fovy range")
    parser.add_argument('--offset', type=float, nargs='*', default=[-0.1, 0, -5.3], help="test camera fovy range")
    parser.add_argument('--ootd', type=bool, default=False, help="use amp mixed precision training")
    parser.add_argument("--sds", action='store_false', help="If reset color and size of the editing points")
    parser.add_argument("--mode", type=str, default="train", help="0:add new object, 1:edit existing object")
    parser.add_argument("--view", type=str, default="all", help="0:add new object, 1:edit existing object")

    opt = parser.parse_args()

    if opt.seed is not None:
        seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # from models.network_3dgaussain import GSNetwork
    # from models.splatting_avatar_model import SplattingAvatarModel
    from models.splatting_cloth import SplattingClothModel
    model = SplattingClothModel(opt, device)
    # model = GPSNetwork(opt)


    valid_dataset = SphericalSamplingGSDataset(opt, device=device, type=opt.mode, H=1024,view = opt.view,
                                             W=768, size=250,sd =True, cloth=opt.cloth)
    valid_loader = valid_dataset.dataloader()

    test_loader = SphericalSamplingGSDataset(opt, device=device, type='test', H=1024,
                                             W=768, size=250,sd =True, cloth=opt.cloth).dataloader()


    guidance=None
    # from models.sd import StableDiffusion

    guidance = StableDiffusion(opt, device, base_path='/home/jian/IDM-VTON/result/checkpoint-4000/')
    # guidance = StableDiffusion(opt, device, base_path='yisol/IDM-VTON')


    # max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
    """
     opt,
     model :SplattingAvatarModel,
     guidance,
     workspace,
     pc_dir,
     dataset,
     """
    trainer = Trainer_SDS(opt, model, guidance, workspace=opt.workspace, pc_dir = opt.data_path, dataset = valid_dataset)

    max_epoch = 2000
    trainer.train(valid_loader, test_loader, max_epoch)
    print("train coarse success")
