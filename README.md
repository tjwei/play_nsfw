# Play_NSFW
This repo contains a few works I played around with 
 [Open NSFW](https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for) from Yahoo, an open source deep learning solution for detecting NSFW( not suitable/safe for work ) images. There are many reasons for an image to be considered as NSFW, but currently open nsfw only considered images with sexual conents.

Most of the stuff requires  theano, lasagne, numpy, PIL and Jupyter notebook.
These works include:
*   theano/lasagne port of Open NSFW
*  Using genetic/evolutionary alogrithm to generate images with high NSFW score
*  Inverse open NSFW: Modified an image to have an arbitrary NSFW score (e.g. 0 to 1 or 0 to 1), but the modification is visually undetectable.

## theano/lasagne port of Open NSFW
[open_nsfw.py](open_nsfw.py) is the main module contains the code of constructing open nsfw network. It parse the deploy.prototxt file from original open nsfw project and construct the corresponding lasagne network layers.
The pretrained weight is stored in nsfw.pkl, which is converted from the caffemodel from original open nsfw project. 
For end users, the simplest way to use it is `from open_nsfw import nsfw_score` and then `nsfw_score(img)`, where img can be the file name of the image or an PIL Image object.  It will return the nsfw_score as a float number between 0 and 1.  An image with socre higher than 0.8 can be cosidered as NSFW. An image with score less than 0.2 can be considered as SFW. 

`nsfw.pkl` and `nsfw_model/deploy.prototxt` needs to be placed in the current working directory. If not, the path of theses files can be passed to `nsfw_score` using keyword arguments `nsfw_pkl_path` and `deploy_prototxt_path`. For example:
```python
nsfw_score(img, 
		nsfw_pkl_path="/home/u/nsfw.pkl", 
		deploy_prototxt_path="/opt/deploy.prototxt")
```
The result of this port is slightly different than the original one (~0.005). One root of the difference is from batch norm layer.  The result of `bn_1` is different up to 1e-7. I might misunderstood the meanings of the blobs of caffe batchnorm layers( I use `mean =blobs[0]/blobs[2]`, `variance=blobs[1]/blobs[2]`, and `epsilon=1e-5`). 
The work is done in jupyter notebook and the `.py` file is converted from [open_nsfw.ipynb](open_nsfw.ipynb). See the notebook for results of example images. 

##  Using genetic/evolutionary alogrithm to generate images with high NSFW score

This part requires [DEAP](https://github.com/DEAP/deap). It use simple and naive evolutionary algorithm to generate images consists with circles or triangles.
The naive and simple algorithm is not particularly efficient, but is interesting to watch it work. When the target is a picture, it can generate recognizable results with some kind of artistic style.
This experiment use the same algorithm but the target function is replaced by NSFW score. We can imagine two possible results: one is probably some kind of sexual related content with artistic style. The other might just some abstract image with high NSFW score (and in artistic style).

It is the later case.


[NSFW-Triangles.ipynb](NSFW-Triangles.ipynb): this is the result for 100 triangles.

![](output/nsfw-triangles.png) 

NSFW score: `0.989984035492` 

[NSFW-Circles.ipynb](NSFW-Circles.ipynb): this is the result for 50 circles.

![](output/nsfw-circle50.png) 

NSFW score: `0.999073386192`

Both images have some what significant portion with skin like colors. This might or might not mean something. Other than this, I haven't see any pattern.

##  Inverse open NSFW

Lasagne and other python packages like scipy provides many efficient optimization method can help us to find an image with certain NSFW score.
We use adam provided by lasagne. Turns out it can find an input data with near to 1 or near to 0 score very quickly. 

Say we start from an NSFW image and run the optimization to get a very low NSFW score.  We are practically trying to find a neighbor of this image having low score. We can imagine two possible scenario. 
* The image is visually unchanged
* Certain part being censored.
It seems that there are images with high scores and images with low scores in every small neighborhood. Since the neighborhood is so small, the change is unnoticeable ny naked eyes. 

However, the problem is, the original Open NSFW uses a preprocess to prepare the input image. It first resize the image to 256x256. And then comressed using JPEG enocing. Then Decompress it then feed to the network.

After applying this procedure, the score of an image can be dramastically change. In order to apply adam or other differential based method to the original image, we need to reformulate the jpeg encoding, decoding process in a differentiable way.  This might take awhile and might not be even practical or possible.

Instead, we construct a covolutional neural network attempt to learn the result image of the jpeg encoding. 
To learn the behaviour of jpeg encoding may need a large network and a lot of time. But we only need the mock jpeg encoder works for certain input and our input are hopefully very similar(in a small neighborhood like mentioned above), so this might just work.

This mock jpeg encoder imporves the result but is not accurate enough. Therefore, we modified the loss function of the mock jpeg encoder, to make it act like an adversary of the inverse nsfw model. By doing so, hopefully the inaccuracy would not help the inverse nfsw model. 

The code is in [Inverse-NSFW.ipynb](Inverse-NSFW.ipynb)
Followings are some results: 
<table>
<tbody>
<tr>
<th>Original</th><th>Inversed</th></tr>
<tr>
<td><img src="output/starry_night-nsfw-0.png" />   <br />
NSFW score: 0.0018048202619
</td>
<td><img src="output/starry_night-nsfw-8000.png" />   <br />
NSFW score: 0.983393967152  
(8000 iterations)
</td>
</tr>
<tr>
<td>
<img src="output/the_scream-nsfw-0.png" />  <br />
NSFW score: 0.00257238722406
</td>
<td>
<img src="output/the_scream-nsfw-8000.png" />  <br />
NSFW score: 0.983393967152
(8000 iterations)
</td>
</tr>
<tr>
<td>
<img src="output/flickr2-nsfw-0.png" />  <br />
NSFW score: 0.952058911324
</td>
<td>
<img src="output/flickr2-nsfw-2000.png" /> <br />
NSFW score: 0.0292308945209 
(2000 iterations)
</td>
</tr>
<tr>
<td>
<img src="output/flickr-nsfw-0.png" />  <br />
NSFW score: 0.951349616051
</td>
<td>
<img src="output/flickr-nsfw-1000.png" />  <br />
NSFW score: 0.0644422993064
(1000 iterations)
</td>
</tr>
</tbody>
</table>

Image license

Under the directory img:
Many image files are from https://github.com/jcjohnson/neural-style and some are public domain

* CC-BY
https://www.flickr.com/photos/two/236883835/in/photolist-mW6j4-mW6kn-Jy6XP-Jy9fR-JxQD8-mW6gp-JxQGc-Jy5Xf-Jy3ES-mW6fr-Jy9eM-mW6hW-JxQqD-JxMdm-mW6nF-JxQtD-JxQxR-mW6mA-JxLZS-Jy72p-JxM2N-Jy3Hh-JxLU7-JxLWb-JxQsz-oH3ixB-4Nri37-JxQJr-JxQFe-Jy3AY-JxMcq-Jy3xG-JxQvT-Jy95i-Jy99n-JxQz2-a97MmE-JxQm6-Jy5V5-a97Nqb-a97PVu-Jy97c-a94ZyK-ADmwE6-B3g6KV-ADmtTV-BysQ61-a97Ptd-a94Zex-dwVdG
ndhu

* CC-BY
https://www.flickr.com/photos/jphotos/5998521863/in/photolist-mW6gp-JxQGc-Jy5Xf-Jy3ES-mW6fr-Jy9eM-mW6hW-JxQqD-JxMdm-mW6nF-JxQtD-JxQxR-mW6mA-JxLZS-Jy72p-JxM2N-Jy3Hh-JxLU7-JxLWb-JxQsz-oH3ixB-4Nri37-JxQJr-JxQFe-Jy3AY-JxMcq-Jy3xG-JxQvT-Jy95i-Jy99n-JxQz2-a97MmE-JxQm6-Jy5V5-a97Nqb-a97PVu-Jy97c-a94ZyK-ADmwE6-B3g6KV-ADmtTV-BysQ61-a97Ptd-a94Zex-dwVdG-fJCm8H-ih4o2w-farbgP-ez8gpV-a244iL
ndhu2

* CC-BY https://www.flickr.com/photos/kurtbudiarto/7257851556/in/photolist-c4mopE-jZZkdx-8bohuZ-bjLvVe-c4wRru-gkp48t-hPNbbL-qfySJz-c84E3h-fBV6ve-aijVu7-fCaHPs-qZ7XnX-f8zn77-eQRUQ5-695MbG-dQNbTm-rowN34-6o4FFz-d3MjtL-9htwN2-r6Yib1-fCau8q-fCaCh9-551xgk-bkueEX-fBVtqt-pbJvac-fCbdY9-aRQ2oH-oUai6n-fCbaRs-fKzB9m-fCbmqA-7kBUGU-eX5Ce9-fBVu9x-ftqE1w-aihyDg-bku5n4-54rawt-nixm7H-dDjykx-55JGBX-c4mnWy-bkudkH-pRMC8L-55JGWa-5A9m5g-6uijiE/
smile face

A few more CC-BY image from flickr used in README.ME and [open_nsfw.ipynb](open_nsfw.ipynb)


* https://www.flickr.com/photos/tucia/3894751161/in/photolist-6WaBQ6-588EVg-qwe3vP-zSLV3m-aA4SSu-9y8bQk-nAmQBE-9dKP36-5eE3sk-ix9v4Y-5xyi5X-ix9QFH-iRbGky-cERRw-abHz5r-cETc2-cxJFo-ofsSfb-4SjUcK-9dKSQ4-8Ks5DP-HE8nP-dHBe6D-6wethg-m13M1-9pSDiL-4XWM5o-b4or4M-97oDt9-qkZDQZ-ix9Gzq-aA5bhb-a6sU1N-ix9m32-aVPgQH-9g5D3o-7zmWKD-dx8m8Z-9dP1tf-7yNMYU-bRxwc-d5cdPb-cERRG-9iQEMN-9dL1Kc-9g5Fi7-5cRoXX-94dX3M-9dNYjw-94h3q9

* https://www.flickr.com/photos/kunshou/132063430/in/photolist-cERRG-9iQEMN-9dL1Kc-9g5Fi7-5cRoXX-94dX3M-9dNYjw-94h3q9-ix9GKW-brsMHh-Hh1XVY-qe7oHB-pQtsm-byXe9T-bm3qLW-768R64-bm3nF3-pxfXAS-jmfdbr-yUpbt-7kMWwA-pyRVUp-dFNmA8-bUsjm5-HZjtsw-7BLvmR-8d9xAT-bUsjns-8CnDLT-8ni3to-q1gEN-iR3aHN-ix9mSi-dDTU34-8CqME1-9pRJdE-6Gvex5-d5h1c7-8RGZzD-C9z6j-pYkRyH-8CnE74-byXifz-a7PfwN-cERRC-fy8rD5-dFTL5j-6uXAhQ-pL1fc-7i7pyq/

* https://www.flickr.com/photos/rallion/92101780/in/photolist-993DJ-oExkuS-cD1q4s-ht2Y-MYiNfa-MYgXfi-MQDy1H-MN6XqK-LW6zT9-4ipCF-p7L2T-7q1W9-NEefq-5wo6zn-74xiZ2-5xYxek-MWnF1e-MtktdN-Mtkmd5/

* https://www.flickr.com/photos/saptel/27120530295/in/photolist-HjxNMV-tMEhp-4963mP-d2fyV7-7rrY29-ePiCT8-8V32rp-Hbbj87-4zXDsd-6eUu8C-bmmaR9-FGFa8-6HYTjR-7tDxUe-Gp3hDo-5vNaDr-aB4WKE-HdvA5B-arZFjR-7f7aDc-nsECZs-4UqPbz-7ijaoR-HjxQ7D-6Qzppw-GUjLCE-4hbZGw-4rzLp6-57seUp-4CoB7v-dvx4qL-6jBvvt-7oKZUu-ndSGh-4dFEac-a8iWP4-4u3DFu-qN4wmN-2vL8H-4dfgqw-XUhyq-4XmiXD-am1kTw-5Rw36b-nXeq97-8oLnz5-aRtaD-cnXUFu-BuQbV-98yiXk

