import os 
import pathlib 
import argparse 
import numpy as np 
import matplotlib .pyplot as plt 

from tqdm import tqdm 
from typing import List 
from skimage import img_as_ubyte 
from skimage .metrics import structural_similarity ,peak_signal_noise_ratio 
from torchmetrics .image .lpip import LearnedPerceptualImagePatchSimilarity 

import torch 
import torch .nn as nn 
import lightning .pytorch as pl 
from torch .utils .data import DataLoader 

from net .AMIFound import AMIFound_Multiexpers 
from net .AMIFound_2 import NestedAMIFound 
from options2 import train_options 
from utils .test_utils import save_img 
from data .dataset_utils_all import IRBenchmarks ,CDD11 




def check_tensor_size (tensor ,name ):
    """检查张量是否超过32位索引限制"""
    max_int32 =2147483647 


    num_elements =tensor .numel ()
    if num_elements >max_int32 :
        print (f"⚠️ 警告: 张量 '{name}' 过大 - 元素数量: {num_elements} > {max_int32}")
        return False 


    for i ,stride in enumerate (tensor .stride ()):
        if stride >max_int32 :
            print (f"⚠️ 警告: 张量 '{name}' 维度 {i} 步长过大: {stride} > {max_int32}")
            return False 

    return True 
def compute_psnr (image_true ,image_test ,image_mask ,data_range =None ):

    err =np .sum ((image_true -image_test )**2 ,dtype =np .float64 )/np .sum (image_mask )
    return 10 *np .log10 ((data_range **2 )/err )


def compute_ssim (tar_img ,prd_img ,cr1 ):
    ssim_pre ,ssim_map =structural_similarity (tar_img ,prd_img ,channel_axis =2 ,gaussian_weights =True ,data_range =1.0 ,
    full =True )
    ssim_map =ssim_map *cr1 
    r =int (3.5 *1.5 +0.5 )
    win_size =2 *r +1 
    pad =(win_size -1 )//2 
    ssim =ssim_map [pad :-pad ,pad :-pad ,:]
    crop_cr1 =cr1 [pad :-pad ,pad :-pad ,:]
    ssim =ssim .sum (axis =0 ).sum (axis =0 )/crop_cr1 .sum (axis =0 ).sum (axis =0 )
    ssim =np .mean (ssim )
    return ssim 


def calc_psnr (img1 ,img2 ,data_range =1.0 ):
    err =np .sum ((img1 -img2 )**2 ,dtype =np .float64 )
    return 10 *np .log10 ((data_range **2 )/(err /img1 .size ))


def calc_ssim (img1 ,img2 ):
    return structural_similarity (img1 ,img2 ,channel_axis =2 ,gaussian_weights =True ,data_range =1.0 ,full =False )




class PLTestModel (pl .LightningModule ):
    def __init__ (self ,opt ):
        super ().__init__ ()


















        self .net =AMIFound_Multiexpers (
        dim =opt .dim ,
        num_blocks =opt .num_blocks ,
        num_dec_blocks =opt .num_dec_blocks ,
        levels =len (opt .num_blocks ),
        heads =opt .heads ,
        num_refinement_blocks =opt .num_refinement_blocks ,
        topk =opt .topk ,
        num_experts =opt .num_exp_blocks ,
        rank =opt .latent_dim ,
        with_complexity =opt .with_complexity ,
        depth_type =opt .depth_type ,
        stage_depth =opt .stage_depth ,
        rank_type =opt .rank_type ,
        complexity_scale =opt .complexity_scale ,)
















    def forward (self ,x ):
        return self .net (x )



def process_large_tensor (net ,tensor ,chunk_size =1 ):
    """分块处理超出32位限制的大张量"""
    try :

        if tensor .dim ()==4 and tensor .size (0 )>1 :
            outputs =[]
            for i in range (0 ,tensor .size (0 ),chunk_size ):
                chunk =tensor [i :i +chunk_size ]
                outputs .append (net (chunk ))
            return torch .cat (outputs ,dim =0 )


        print ("尝试按空间维度分块...")
        return spatial_chunk_processing (net ,tensor )

    except RuntimeError as e :
        print (f"分块处理失败: {str(e)}")
        return None 


def spatial_chunk_processing (net ,tensor ,tile_size =512 ):
    """按空间维度分块处理图像张量 (当h或w > 1800时自动调用)"""
    if tensor .dim ()!=4 :
        print ("空间分块仅支持4D张量 [B,C,H,W]")
        return net (tensor )

    B ,C ,H ,W =tensor .shape 


    actual_tile_size =min (tile_size ,H ,W )


    num_tiles_h =(H +actual_tile_size -1 )//actual_tile_size 
    num_tiles_w =(W +actual_tile_size -1 )//actual_tile_size 


    overlap =min (32 ,actual_tile_size //4 )


    output =torch .zeros_like (tensor )

    for b in range (B ):
        for i in range (num_tiles_h ):
            for j in range (num_tiles_w ):

                h_start =max (0 ,i *actual_tile_size -overlap )
                h_end =min (H ,(i +1 )*actual_tile_size +overlap )
                w_start =max (0 ,j *actual_tile_size -overlap )
                w_end =min (W ,(j +1 )*actual_tile_size +overlap )


                tile =tensor [b :b +1 ,:,h_start :h_end ,w_start :w_end ]


                processed_tile =net (tile )
                if isinstance (processed_tile ,(list ,tuple )):
                    processed_tile =processed_tile [0 ]


                crop_top =overlap if i >0 else 0 
                crop_bottom =processed_tile .size (2 )-(overlap if i <num_tiles_h -1 else 0 )
                crop_left =overlap if j >0 else 0 
                crop_right =processed_tile .size (3 )-(overlap if j <num_tiles_w -1 else 0 )


                if crop_bottom <=crop_top or crop_right <=crop_left :
                    continue 

                valid_region =processed_tile [:,:,crop_top :crop_bottom ,crop_left :crop_right ]


                out_h_start =i *actual_tile_size 
                out_h_end =min (H ,(i +1 )*actual_tile_size )
                out_w_start =j *actual_tile_size 
                out_w_end =min (W ,(j +1 )*actual_tile_size )


                valid_h =out_h_end -out_h_start 
                valid_w =out_w_end -out_w_start 

                if valid_region .size (2 )!=valid_h or valid_region .size (3 )!=valid_w :

                    valid_region =torch .nn .functional .interpolate (
                    valid_region ,size =(valid_h ,valid_w ),mode ='bilinear',align_corners =False 
                    )


                output [b :b +1 ,:,out_h_start :out_h_end ,out_w_start :out_w_end ]=valid_region 

    return output 

def run_test (opts ,net ,dataset ,factor =8 ):
    testloader =DataLoader (dataset ,batch_size =1 ,pin_memory =True ,shuffle =False ,drop_last =False ,num_workers =16 )

    if opts .save_results :
        pathlib .Path (os .path .join (os .getcwd (),f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}")).mkdir (
        parents =True ,exist_ok =True )
    calc_lpips =LearnedPerceptualImagePatchSimilarity (net_type ='vgg',normalize =True ,reduction ="mean").cuda ()
    psnr ,ssim ,lpips =[],[],[]
    with torch .no_grad ():
        for ([clean_name ,de_id ],degrad_patch ,clean_patch )in tqdm (testloader ):
            degrad_patch ,clean_patch =degrad_patch .cuda (),clean_patch .cuda ()


            _ ,_ ,h ,w =degrad_patch .shape 
            if h >2000 or w >2000 :
                print (f"⚠️ 图像尺寸过大 ({h}x{w})，启动分块处理")
                restored =spatial_chunk_processing (net ,degrad_patch )
            else :
                restored =net (degrad_patch )


            if isinstance (restored ,List )and len (restored )==2 :
                restored ,_ =restored 


            assert restored .shape ==clean_patch .shape ,"Restored and clean patch shape mismatch."


            restored =torch .clamp (restored ,0 ,1 )
            lpips .append (calc_lpips (clean_patch ,restored ).cpu ().numpy ())

            restored =restored .cpu ().detach ().permute (0 ,2 ,3 ,1 ).squeeze (0 ).numpy ()
            degrad_patch =degrad_patch .cpu ().detach ().permute (0 ,2 ,3 ,1 ).squeeze (0 ).numpy ()
            clean =clean_patch .cpu ().detach ().permute (0 ,2 ,3 ,1 ).squeeze (0 ).numpy ()
            ssim .append (calc_ssim (clean ,restored ))
            psnr_temp =peak_signal_noise_ratio (clean ,restored ,data_range =1 )
            psnr .append (psnr_temp )

            if opts .save_results :
                save_name =os .path .splitext (os .path .split (clean_name [0 ])[-1 ])[0 ]+'_'+str (
                round (psnr_temp ,2 ))+'.png'
                save_img (
                (os .path .join (os .getcwd (),
                f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}",
                save_name )),
                img_as_ubyte (restored ))

    print ('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format (np .mean (psnr ),np .mean (ssim ),np .mean (lpips )))



def run_synllie (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )



def run_gopro (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )



def run_derain (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )



def run_dehaze (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )



def run_denoise_15 (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )


def run_denoise_25 (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )


def run_denoise_50 (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_denoise (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_deblur (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_cdd11 (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_CT (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_MR (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_Endoscopy (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_Fundus (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_PET (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_Ultrasound (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

def run_X_ray (opts ,net ,dataset ,factor =8 ):
    run_test (opts ,net ,dataset ,factor )

RUNNERS ={
"Endoscopy":run_Endoscopy ,
"Fundus":run_Fundus ,
"PET":run_PET ,
"Ultrasound":run_Ultrasound ,
"X-ray":run_X_ray ,
"CT":run_CT ,
"MR":run_MR ,

"synllie":run_Endoscopy ,
"deblur":run_Fundus ,
"derain":run_PET ,
"dehaze":run_Ultrasound ,
"denoise":run_X_ray ,
}



def main (opt ):
    np .random .seed (0 )
    torch .manual_seed (0 )
    torch .cuda .manual_seed (0 )


    net =PLTestModel .load_from_checkpoint (
    os .path .join (opt .ckpt_dir ,opt .checkpoint_id ,"last.ckpt"),opt =opt ).cuda ()
    net .eval ()
    for de in opt .benchmarks :
        ind_opt =opt 
        ind_opt .benchmarks =[de ]

        if "CDD11"in opt .trainset :
            _ ,subset =opt .trainset .split ("_",maxsplit =1 )
            dataset =CDD11 (opt ,split ="test",subset =subset )
        else :
            dataset =IRBenchmarks (ind_opt )

        print ("--------> Testing on",de ,"testset.")
        print ("\n")
        if de not in RUNNERS :
            raise NotImplementedError (f"Unsupported benchmark/modality: {de}")
        RUNNERS [de ](opt ,net ,dataset ,factor =8 )


def depth_type (value ):
    try :
        return int (value )
    except ValueError :
        return value 


def str2bool (v ):
    if isinstance (v ,bool ):
        return v 
    if v .lower ()in ('yes','true','t','y','1'):
        return True 
    elif v .lower ()in ('no','false','f','n','0'):
        return False 
    else :
        raise argparse .ArgumentTypeError ('Boolean value expected.')


if __name__ =='__main__':
    train_opt =train_options ()

    main (train_opt )
