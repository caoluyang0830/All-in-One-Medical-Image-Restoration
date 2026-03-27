import os 
import cv2 
import glob 
import random 
import numpy as np 
from PIL import Image 

from torch .utils .data import Dataset 
from torchvision .transforms import ToPILImage ,Compose ,RandomCrop ,ToTensor ,Resize ,InterpolationMode 

from data .degradation_utils import Degradation 
from utils .image_utils import random_augmentation ,crop_img 


class CDD11 (Dataset ):
    def __init__ (self ,args ,split :str ="train",subset :str ="all"):
        super (CDD11 ,self ).__init__ ()

        self .args =args 
        self .toTensor =ToTensor ()
        self .de_type =self .args .de_type 
        self .dataset_split =split 
        self .subset =subset 
        if split =="train":
            self .patch_size =args .patch_size 
        else :
            self .patch_size =64 

        self ._init ()

    def __getitem__ (self ,index ):

        if self .dataset_split =="train":
            degradation_type =random .choice (list (self .degraded_dict .keys ()))
            degraded_image_path =random .choice (self .degraded_dict [degradation_type ])
        else :
            degradation_type =self .subset 
            degraded_image_path =self .degraded_dict [degradation_type ][index ]



        degraded_name =os .path .basename (degraded_image_path )


        image_name =os .path .basename (degraded_image_path )
        assert degraded_name ==image_name 
        clean_image_path =os .path .join (os .path .dirname (self .clean [0 ]),image_name )



        lr =np .array (Image .open (degraded_image_path ).convert ('RGB'))

        hr =np .array (Image .open (clean_image_path ).convert ('RGB'))

        if self .dataset_split =="train":
            lr ,hr =random_augmentation (*self ._crop_patch (lr ,hr ))


        lr =self .toTensor (lr )
        hr =self .toTensor (hr )

        return [clean_image_path ,degradation_type ],lr ,hr 

    def __len__ (self ):
        return sum (len (images )for images in self .degraded_dict .values ())

    def _init (self ):
        data_dir =os .path .join (self .args .data_file_dir ,"cdd11")
        self .clean =sorted (glob .glob (os .path .join (data_dir ,f"{self.dataset_split}/clear","*.png")))

        if len (self .clean )==0 :
            raise ValueError (f"No clean images found in {os.path.join(data_dir, f'{self.dataset_split}/clear')}")

        self .degraded_dict ={}
        allowed_degradation_folders =self ._filter_degradation_folders (data_dir )
        for folder in allowed_degradation_folders :
            folder_name =os .path .basename (folder .strip ('/'))
            degraded_images =sorted (glob .glob (os .path .join (folder ,"*.png")))

            if len (degraded_images )==0 :
                raise ValueError (f"No images found in {folder_name}")


            if self .dataset_split =="train":
                degraded_images *=2 

            self .degraded_dict [folder_name ]=degraded_images 

    def _filter_degradation_folders (self ,data_dir ):
        """
        This function returns folders based on the degradation_type_mode.
        'single', 'double', 'triple', or 'all' degradation types will be returned.
        """
        degradation_folders =sorted (glob .glob (os .path .join (data_dir ,self .dataset_split ,"*/")))
        filtered_folders =[]

        for folder in degradation_folders :
            folder_name =os .path .basename (folder .strip ('/'))
            if folder_name =="clear":
                continue 


            degradation_count =folder_name .count ('_')+1 


            if self .subset =="single"and degradation_count ==1 :
                filtered_folders .append (folder )
            elif self .subset =="double"and degradation_count ==2 :
                filtered_folders .append (folder )
            elif self .subset =="triple"and degradation_count ==3 :
                filtered_folders .append (folder )
            elif self .subset =="all":
                filtered_folders .append (folder )

            elif self .subset not in ["single","double","triple","all"]:
                if folder_name ==self .subset :
                    filtered_folders .append (folder )

        print (f"Degradation type mode: {self.subset}")
        print (f"Loading degradation folders: {[os.path.basename(f.strip('/')) for f in filtered_folders]}")
        return filtered_folders 

    def _crop_patch (self ,img_1 ,img_2 ):
        H ,W =img_1 .shape [:2 ]



        if H <self .args .patch_size or W <self .args .patch_size :








            scale =max (self .args .patch_size /H ,self .args .patch_size /W )
            new_H ,new_W =int (H *scale ),int (W *scale )
            img_1 =cv2 .resize (img_1 ,(new_W ,new_H ),interpolation =cv2 .INTER_LINEAR )
            img_2 =cv2 .resize (img_2 ,(new_W ,new_H ),interpolation =cv2 .INTER_LINEAR )
            H ,W =new_H ,new_W 


        ind_H =random .randint (0 ,H -self .args .patch_size )
        ind_W =random .randint (0 ,W -self .args .patch_size )


        patch_1 =img_1 [ind_H :ind_H +self .args .patch_size ,ind_W :ind_W +self .args .patch_size ]
        patch_2 =img_2 [ind_H :ind_H +self .args .patch_size ,ind_W :ind_W +self .args .patch_size ]

        return patch_1 ,patch_2 


class AIOTrainDataset (Dataset ):
    """
    Dataset class for training on degraded images (支持CT/MRI按Epoch动态跨步采样).
    """

    def __init__ (self ,args ):
        super (AIOTrainDataset ,self ).__init__ ()
        self .args =args 
        self .de_temp =0 
        self .de_type =self .args .de_type 
        self .D =Degradation (args )
        self .de_dict ={dataset :idx for idx ,dataset in enumerate (self .de_type )}
        self .de_dict_reverse ={idx :dataset for idx ,dataset in enumerate (self .de_type )}


        self .crop_transform =Compose ([
        ToPILImage (),
        RandomCrop (args .patch_size ),
        ])
        self .toTensor =ToTensor ()


        self .total_epochs =args .epochs 
        self .current_epoch =0 


        self ._init_lr ()

        self .current_lr =[]
        self .current_hr =[]

    def __getitem__ (self ,idx ):
        """获取单个样本（从当前Epoch的动态样本列表中读取）"""

        lr_sample =self .current_lr [idx ]
        de_id =lr_sample ["de_type"]
        deg_type =self .de_dict_reverse [de_id ]

        if deg_type in ["denoise_15","denoise_25","denoise_50"]:

            hr =crop_img (np .array (Image .open (lr_sample ["img"]).convert ('RGB')),base =16 )
            hr =self .crop_transform (hr )
            hr =np .array (hr )
            hr =random_augmentation (hr )[0 ]
            lr =self .D .single_degrade (hr ,de_id )
        else :
            if deg_type =="dehaze_original":

                lr =crop_img (np .array (Image .open (lr_sample ["img"]).convert ('RGB')),base =16 )
                clean_name =self ._get_nonhazy_name (lr_sample ["img"])
                hr =crop_img (np .array (Image .open (clean_name ).convert ('RGB')),base =16 )
            else :

                hr_sample =self .current_hr [idx ]
                lr =crop_img (np .array (Image .open (lr_sample ["img"]).convert ('RGB')),base =16 )
                hr =crop_img (np .array (Image .open (hr_sample ["img"]).convert ('RGB')),base =16 )


            lr ,hr =random_augmentation (*self ._crop_patch (lr ,hr ))


        lr =self .toTensor (lr )
        hr =self .toTensor (hr )

        return [lr_sample ["img"],de_id ],lr ,hr 

    def __len__ (self ):
        """当前Epoch的总样本数（动态变化）"""
        return len (self .current_lr )

    def set_epoch (self ,current_epoch ):
        """
        关键方法：更新当前Epoch，动态生成CT/MRI样本子集
        参数：current_epoch - 当前训练的Epoch（从0开始）
        """
        self .current_epoch =current_epoch 


        self ._generate_current_ct_samples ()

        self ._generate_current_mr_samples ()

        self ._merge_current_tasks ()

    def _init_lr (self ):
        """初始化所有数据集（CT/MRI保存完整训练对，其他数据集生成静态样本）"""

        if 'Endoscopy'in self .de_type :
            self ._init_synllie (id =self .de_dict ['Endoscopy'])
        if 'Fundus'in self .de_type :
            self ._init_deblur (id =self .de_dict ['Fundus'])
        if 'PET'in self .de_type :
            self ._init_derain (id =self .de_dict ['PET'])
        if 'Ultrasound'in self .de_type :
            self ._init_dehaze (id =self .de_dict ['Ultrasound'])
        if 'X-ray'in self .de_type :
            self ._init_denoise (id =self .de_dict ['X-ray'])
        if 'CT'in self .de_type :
            self ._init_CT (id =self .de_dict ['CT'])
        if 'MR'in self .de_type :
            self ._init_MR (id =self .de_dict ['MR'])
        if 'denoise_15'in self .de_type :
            self ._init_clean (id =0 )
        if 'denoise_25'in self .de_type :
            self ._init_clean (id =0 )
        if 'denoise_50'in self .de_type :
            self ._init_clean (id =0 )

    def _merge_current_tasks (self ):
        """合并静态数据集（如Endoscopy）与当前Epoch的CT/MRI动态样本"""
        self .current_lr =[]
        self .current_hr =[]


        if hasattr (self ,'synllie_lr')and hasattr (self ,'synllie_hr'):
            self .current_lr +=self .synllie_lr 
            self .current_hr +=self .synllie_hr 
        if hasattr (self ,'denoise_lr')and hasattr (self ,'denoise_hr'):
            self .current_lr +=self .denoise_lr 
            self .current_hr +=self .denoise_hr 
        if hasattr (self ,'s15_ids'):
            self .current_lr +=self .s15_ids 
            self .current_hr +=self .s15_ids 
        if hasattr (self ,'s25_ids'):
            self .current_lr +=self .s25_ids 
            self .current_hr +=self .s25_ids 
        if hasattr (self ,'s50_ids'):
            self .current_lr +=self .s50_ids 
            self .current_hr +=self .s50_ids 
        if hasattr (self ,'deblur_lr')and hasattr (self ,'deblur_hr'):
            self .current_lr +=self .deblur_lr 
            self .current_hr +=self .deblur_hr 
        if hasattr (self ,'derain_lr')and hasattr (self ,'derain_hr'):
            self .current_lr +=self .derain_lr 
            self .current_hr +=self .derain_hr 
        if hasattr (self ,'dehaze_lr')and hasattr (self ,'dehaze_hr'):
            self .current_lr +=self .dehaze_lr 
            self .current_hr +=self .dehaze_hr 


        if hasattr (self ,'current_CT_lr')and hasattr (self ,'current_CT_hr'):
            self .current_lr +=self .current_CT_lr 
            self .current_hr +=self .current_CT_hr 


        if hasattr (self ,'current_MR_lr')and hasattr (self ,'current_MR_hr'):
            self .current_lr +=self .current_MR_lr 
            self .current_hr +=self .current_MR_hr 


        print (f"Epoch {self.current_epoch} - Total samples: {len(self.current_lr)} "
        f"(CT: {len(self.current_CT_lr) if hasattr(self, 'current_CT_lr') else 0}, "
        f"MR: {len(self.current_MR_lr) if hasattr(self, 'current_MR_lr') else 0})")

    def _generate_current_ct_samples (self ):
        """生成当前Epoch的CT样本子集（跨步采样）"""
        if not hasattr (self ,'ct_train_pairs'):
            self .current_CT_lr =[]
            self .current_CT_hr =[]
            return 


        total =self .total_ct_train 
        batch_size =self .batch_ct 
        start_idx =self .current_epoch *batch_size 


        if self .current_epoch ==self .total_epochs -1 :
            end_idx =total 
        else :
            end_idx =start_idx +batch_size 


        current_pairs =self .ct_train_pairs [start_idx :end_idx ]

        self .current_CT_lr =[{"img":pair [0 ],"de_type":self .ct_de_type_id }for pair in current_pairs ]
        self .current_CT_hr =[{"img":pair [1 ],"de_type":self .ct_de_type_id }for pair in current_pairs ]

    def _generate_current_mr_samples (self ):
        """生成当前Epoch的MRI样本子集（与CT逻辑一致）"""
        if not hasattr (self ,'mr_train_pairs'):
            self .current_MR_lr =[]
            self .current_MR_hr =[]
            return 

        total =self .total_mr_train 
        batch_size =self .batch_mr 
        start_idx =self .current_epoch *batch_size 

        if self .current_epoch ==self .total_epochs -1 :
            end_idx =total 
        else :
            end_idx =start_idx +batch_size 

        current_pairs =self .mr_train_pairs [start_idx :end_idx ]
        self .current_MR_lr =[{"img":pair [0 ],"de_type":self .mr_de_type_id }for pair in current_pairs ]
        self .current_MR_hr =[{"img":pair [1 ],"de_type":self .mr_de_type_id }for pair in current_pairs ]




    def _init_CT (self ,id ):
        """修改：保存完整CT训练对，计算每个Epoch的采样量"""
        random .seed (42 )
        inputs ="/data1/luyang/data/extracted_top50_samples/CT_metal_artifacts"
        targets ="/data1/luyang/data/extracted_top50_samples/CT"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))
        if len (lr_files )!=len (hr_files ):
            print (f"警告: CT文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")
        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        self .ct_train_pairs =paired_files [:split_index ]
        self .ct_de_type_id =id 
        self .total_ct_train =len (self .ct_train_pairs )


        self .batch_ct =self .total_ct_train //self .total_epochs 
        self .ct_remainder =self .total_ct_train %self .total_epochs 


        print (f"CT - Total training pairs: {self.total_ct_train}, "
        f"Batch per epoch: {self.batch_ct}, "
        f"Remainder: {self.ct_remainder}")

    def _init_MR (self ,id ):
        """修改：保存完整MRI训练对，计算每个Epoch的采样量（与CT逻辑一致）"""
        random .seed (42 )
        inputs ="/data1/luyang/data/extracted_top50_samples/MR_LQ"
        targets ="/data1/luyang/data/extracted_top50_samples/MR"

        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))
        if len (lr_files )!=len (hr_files ):
            print (f"警告: MR文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")
        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )

        split_index =int (len (paired_files )*0.7 )
        self .mr_train_pairs =paired_files [:split_index ]
        self .mr_de_type_id =id 
        self .total_mr_train =len (self .mr_train_pairs )

        self .batch_mr =self .total_mr_train //self .total_epochs 
        self .mr_remainder =self .total_mr_train %self .total_epochs 

        print (f"MR - Total training pairs: {self.total_mr_train}, "
        f"Batch per epoch: {self.batch_mr}, "
        f"Remainder: {self.mr_remainder}")

    def _init_synllie (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/Endoscopy_dark"
        targets ="/data1/luyang/data/extracted_top50_samples/Endoscopy"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        train_pairs =paired_files [:split_index ]


        train_lr =[pair [0 ]for pair in train_pairs ]
        train_hr =[pair [1 ]for pair in train_pairs ]

        self .synllie_lr =[{"img":x ,"de_type":id }for x in train_lr ]
        self .synllie_hr =[{"img":x ,"de_type":id }for x in train_hr ]

        self .synllie_counter =0 
        print ("Total Endoscopy training pairs : {}".format (len (self .synllie_lr )))
        print ("Repeated Dataset length : {}".format (len (self .synllie_hr )))

    def _init_deblur (self ,id ):
        """ Initialize the GoPro training dataset with 7:3 split """

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/Fundus_spot_light"
        targets ="/data1/luyang/data/extracted_top50_samples/Fundus"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        train_pairs =paired_files [:split_index ]


        train_lr =[pair [0 ]for pair in train_pairs ]
        train_hr =[pair [1 ]for pair in train_pairs ]

        self .deblur_lr =[{"img":x ,"de_type":id }for x in train_lr ]
        self .deblur_hr =[{"img":x ,"de_type":id }for x in train_hr ]

        self .deblur_counter =0 
        print ("Total Fundus training pairs : {}".format (len (self .deblur_lr )))
        print ("Repeated Dataset length : {}".format (len (self .deblur_hr )))

    def _init_derain (self ,id ):
        """ Initialize the deraining dataset with 7:3 split """

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/PET_denoised"
        targets ="/data1/luyang/data/extracted_top50_samples/PET"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        train_pairs =paired_files [:split_index ]


        train_lr =[pair [0 ]for pair in train_pairs ]
        train_hr =[pair [1 ]for pair in train_pairs ]

        self .derain_lr =[{"img":x ,"de_type":id }for x in train_lr ]
        self .derain_hr =[{"img":x ,"de_type":id }for x in train_hr ]

        self .derain_counter =0 
        print ("Total PET training pairs : {}".format (len (self .derain_lr )))
        print ("Repeated Dataset length : {}".format (len (self .derain_hr )))
    def _init_dehaze (self ,id ):
        """ Initialize the deraining dataset with 7:3 split """

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/Ultrasound_sound_artifacts"
        targets ="/data1/luyang/data/extracted_top50_samples/Ultrasound"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        train_pairs =paired_files [:split_index ]


        train_lr =[pair [0 ]for pair in train_pairs ]
        train_hr =[pair [1 ]for pair in train_pairs ]

        self .dehaze_lr =[{"img":x ,"de_type":id }for x in train_lr ]
        self .dehaze_hr =[{"img":x ,"de_type":id }for x in train_hr ]

        self .dehaze_counter =0 
        print ("Total Ultrasound training pairs : {}".format (len (self .dehaze_lr )))
        print ("Repeated Dataset length : {}".format (len (self .dehaze_hr )))















    def _init_denoise (self ,id ):
        if 'X-ray'in self .de_type :
            random .seed (42 )


            inputs ="/data1/luyang/data/extracted_top50_samples/X_ray_blur"
            targets ="/data1/luyang/data/extracted_top50_samples/X_ray"


            lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
            hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


            if len (lr_files )!=len (hr_files ):
                print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


            paired_files =list (zip (lr_files ,hr_files ))
            random .shuffle (paired_files )


            split_index =int (len (paired_files )*0.7 )
            train_pairs =paired_files [:split_index ]


            train_lr =[pair [0 ]for pair in train_pairs ]
            train_hr =[pair [1 ]for pair in train_pairs ]

            self .denoise_lr =[{"img":x ,"de_type":id }for x in train_lr ]
            self .denoise_hr =[{"img":x ,"de_type":id }for x in train_hr ]

            self .denoise_counter =0 
            print ("Total X_ray training pairs : {}".format (len (self .denoise_lr )))
            print ("Repeated Dataset length : {}".format (len (self .denoise_hr )))


























































    def _crop_patch (self ,img_1 ,img_2 ):
        H ,W =img_1 .shape [:2 ]



        if H <self .args .patch_size or W <self .args .patch_size :








            scale =max (self .args .patch_size /H ,self .args .patch_size /W )
            new_H ,new_W =int (H *scale ),int (W *scale )
            img_1 =cv2 .resize (img_1 ,(new_W ,new_H ),interpolation =cv2 .INTER_LINEAR )
            img_2 =cv2 .resize (img_2 ,(new_W ,new_H ),interpolation =cv2 .INTER_LINEAR )
            H ,W =new_H ,new_W 


        ind_H =random .randint (0 ,H -self .args .patch_size )
        ind_W =random .randint (0 ,W -self .args .patch_size )


        patch_1 =img_1 [ind_H :ind_H +self .args .patch_size ,ind_W :ind_W +self .args .patch_size ]
        patch_2 =img_2 [ind_H :ind_H +self .args .patch_size ,ind_W :ind_W +self .args .patch_size ]

        return patch_1 ,patch_2 

    def _get_nonhazy_name (self ,hazy_name ):
        dir_name =os .path .dirname (os .path .dirname (hazy_name ))+"/clear"
        name =hazy_name .split ('/')[-1 ].split ('_')[0 ]
        suffix =os .path .splitext (hazy_name )[1 ]
        nonhazy_name =dir_name +"/"+name +suffix 
        return nonhazy_name 


class IRBenchmarks (Dataset ):
    def __init__ (self ,args ):
        super (IRBenchmarks ,self ).__init__ ()

        self .args =args 
        self .benchmarks =args .benchmarks 
        self .de_type =self .args .de_type 
        self .de_dict ={dataset :idx for idx ,dataset in enumerate (self .de_type )}

        self .toTensor =ToTensor ()

        self .resize =Resize (size =(512 ,512 ),interpolation =InterpolationMode .NEAREST )

        self ._init_lr ()

    def __getitem__ (self ,idx ):
        lr_sample =self .lr [idx ]
        de_id =lr_sample ["de_type"]

        if "denoise_15"in self .benchmarks or "denoise_25"in self .benchmarks or "denoise_50"in self .benchmarks or "denoise_100"in self .benchmarks or "denoise_75"in self .benchmarks :
            sigma =int (self .benchmarks [-1 ].split ("_")[-1 ])
            hr =crop_img (np .array (Image .open (lr_sample ["img"]).convert ('RGB')),base =16 )
            lr ,_ =self ._add_gaussian_noise (hr ,sigma )
        else :
            hr_sample =self .hr [idx ]
            lr =crop_img (np .array (Image .open (lr_sample ["img"]).convert ('RGB')),base =16 )
            hr =crop_img (np .array (Image .open (hr_sample ["img"]).convert ('RGB')),base =16 )

        lr =self .toTensor (lr )
        hr =self .toTensor (hr )
        return [lr_sample ["img"],de_id ],lr ,hr 

    def __len__ (self ):
        return len (self .lr )

    def _init_lr (self ):










        if 'Endoscopy'in self .de_type :
            self ._init_synllie (id =self .de_dict ['Endoscopy'])
        if 'Fundus'in self .de_type :
            self ._init_deblurring (id =self .de_dict ['Fundus'])
        if 'PET'in self .de_type :
            self ._init_derain (id =self .de_dict ['PET'])
        if 'Ultrasound'in self .de_type :
            self ._init_dehaze (id =self .de_dict ['Ultrasound'])
        if 'X-ray'in self .de_type :
            self ._init_denoise (id =self .de_dict ['X-ray'])
        if 'CT'in self .de_type :
            self ._init_CT (id =self .de_dict ['CT'])
        if 'MR'in self .de_type :
            self ._init_MR (id =self .de_dict ['MR'])







    def _get_nonhazy_name (self ,hazy_name ):
        dir_name =os .path .dirname (os .path .dirname (hazy_name ))+"/gt"
        name =hazy_name .split ('/')[-1 ].split ('_')[0 ]
        suffix =os .path .splitext (hazy_name )[1 ]
        nonhazy_name =dir_name +"/"+name +'.png'
        return nonhazy_name 

    def _add_gaussian_noise (self ,clean_patch ,sigma ):
        noise =np .random .randn (*clean_patch .shape )
        noisy_patch =np .clip (clean_patch +noise *sigma ,0 ,255 ).astype (np .uint8 )
        return noisy_patch ,clean_patch 













    def _init_CT (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/CT_metal_artifacts"
        targets ="/data1/luyang/data/extracted_top50_samples/CT"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]

        print ("Total CT testing pairs : {}".format (len (self .hr )))
    def _init_MR (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/MR_LQ"
        targets ="/data1/luyang/data/extracted_top50_samples/MR"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]

        print ("Total CT testing pairs : {}".format (len (self .hr )))
    def _init_synllie (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/Endoscopy_dark"
        targets ="/data1/luyang/data/extracted_top50_samples/Endoscopy"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]

        print ("Total LLIE testing pairs : {}".format (len (self .hr )))
    def _init_deblurring (self ,id ):


        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/Fundus_spot_light"
        targets ="/data1/luyang/data/extracted_top50_samples/Fundus"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]

        print ("Total LLIE testing pairs : {}".format (len (self .hr )))
    def _init_derain (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/PET_denoised"
        targets ="/data1/luyang/data/extracted_top50_samples/PET"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]



        print ("Total LLIE testing pairs : {}".format (len (self .hr )))
    def _init_dehaze (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/Ultrasound_sound_artifacts"
        targets ="/data1/luyang/data/extracted_top50_samples/Ultrasound"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]

        print ("Total LLIE testing pairs : {}".format (len (self .hr )))
    def _init_denoise (self ,id ):

        random .seed (42 )


        inputs ="/data1/luyang/data/extracted_top50_samples/X_ray_blur"
        targets ="/data1/luyang/data/extracted_top50_samples/X_ray"


        lr_files =sorted (glob .glob (os .path .join (inputs ,"*.png")))
        hr_files =sorted (glob .glob (os .path .join (targets ,"*.png")))


        if len (lr_files )!=len (hr_files ):
            print (f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")


        paired_files =list (zip (lr_files ,hr_files ))
        random .shuffle (paired_files )


        split_index =int (len (paired_files )*0.7 )
        test_pairs =paired_files [split_index :]


        test_lr =[pair [0 ]for pair in test_pairs ]
        test_hr =[pair [1 ]for pair in test_pairs ]


        self .lr =[{"img":x ,"de_type":id }for x in test_lr ]
        self .hr =[{"img":x ,"de_type":id }for x in test_hr ]

        print ("Total LLIE testing pairs : {}".format (len (self .hr )))





































































