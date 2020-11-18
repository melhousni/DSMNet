## DSMNet
This repo contains the code and files necessary to reproduce the results published in our paper 'Height Prediction and Refinement from Aerial Images with Semantic and Geometric Guidance'. Our method relies on a two stage pipeline : First, a multi-task network is used to predict the height, semantic labels and surface normals of an input RGB aerial image. Next, we use a denoising autoencoder to refine our height prediction in order to produce higher quality height maps.

![](/images/fullnet.png)

## Network
### Prerequisites  

### Testing
Both datasets can be download here. The data was organized and seperated into tiles to speed up the training process. No further pre-processing was done.  
Our checkpoints can be found here.  

When unzipping the datasets and checkpoints, make sure to respect the following folder structure :  
  
root  
-datasets  
--DFC2018  
---RGB  
---SEM  
---DSM  
---DEM  
--Vaihingen  
---RGB  
---SEM  
---NDSM  
-checkpoints  
--DFC2018  
--Vaihingen  

Next step is to use the test_dsm.py script to test the prediction and refinement networks by using :  
**python test_dsm.py [dataset] [refinement_flag]**  
For example, to test the results of the prediction and refinement networks combined on the DFC2018 dataset, use :  
python test_dsm.py DFC2018 True  
To test the results of the prediction network onlys combined on the Vaihingen dataset, use :  
python test_dsm.py Vaihingen False  
The output files will be saved to the /output folder.  

### Training
To train your own MTL prediction network, use:  
**python train_mtl.py [dataset]**  
For example, to train the MTL prediction network on the DFC2018 dataset, use :    
python train_mtl.py DFC2018   

<img src="/images/mtl_output.png" width="500" height="400"/>  

To train your own refinement network, first you'll need a checkpoint for the MT prediction network, then you can use:  
**python train_ec.py [dataset]**    
For example, to train therefinement network on the Vaihingen dataset, use :    
python train_ec.py Vaihingen   

<img src="/images/refinement_output.png" width="500" height="400"/>

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{du2020dh3d,
        title={DH3D: Deep Hierarchical 3D Descriptors for Robust Large-Scale 6DoF Relocalization},
        author={Du, Juan and Wang, Rui and Cremers, Daniel},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2020}
    }



