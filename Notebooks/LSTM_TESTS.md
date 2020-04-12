# Notebooks explanation
Right now, I am not using the whole dataset in order to speed up this minor tests, when we decide the final model I will use the full dataset. I am currently using 10 folders out of 24 (which, if we count 2 signers per recording in a folder means 20 out of 48 videos). 
## List
First, to train with just one signer, I use signer with id "0" in folders 190425_aslX.
- [**OneSingle_kp**](../Notebooks/LSTM_2D_to_3D-OneSingle_kp.ipynb): Training for a single keypoint estimation (in this case body left shoulder keypoint) with videos from just one signer. 
  - Using: 6 folders; 6 videos; 1 signer.

- [**One**](../Notebooks/LSTM_2D_to_3D-One.ipynb): Training for the whole keypoint set estimation with videos from just one signer. 
  - Using: 6 folders; 6 videos; 1 signer.

Then, I train with all signers in the 10 folders mentioned above.
- [**LotSingle_kp**](../Notebooks/LSTM_2D_to_3D-LotSingle_kp.ipynb): Training for a single keypoint estimation (in this case body left shoulder keypoint) with videos from 5 different signers. 
  - Using: 10 folders; 20 videos; 5 signers (3 videos, 3 videos, 6 videos, 5 videos and 3 videos, respectively).

- [**Lot**](../Notebooks/LSTM_2D_to_3D-Lot.ipynb): Training for the whole keypoint set estimation with videos from 5 different signers. 
  - Using: 10 folders; 20 videos; 5 signers (3 videos, 3 videos, 6 videos, 5 videos and 3 videos, respectively).

Finally, I created training notebooks with estimation for either just face keypoints, hands keypoints or body keypoints.
- [**Face**](../Notebooks/LSTM_2D_to_3D-Face.ipynb): Training for face keypoints estimation with videos from 5 different signers.
  - Using: 10 folders; 20 videos; 5 signers (3 videos, 3 videos, 6 videos, 5 videos and 3 videos, respectively).

- [**Hands**](../Notebooks/LSTM_2D_to_3D-Hands.ipynb): Training for hands keypoints estimation with videos from 5 different signers.
  - Using: 10 folders; 20 videos; 5 signers (3 videos, 3 videos, 6 videos, 5 videos and 3 videos, respectively).

- [**Body**](../Notebooks/LSTM_2D_to_3D-Body.ipynb): Training for body keypoints estimation with videos from 5 different signers.
  - Using: 10 folders; 20 videos; 5 signers (3 videos, 3 videos, 6 videos, 5 videos and 3 videos, respectively).
  
## Metric
The metric I use to evaluate the performance is the Mean Per Joint Position Error (MPJPE). It gives the mean Euclidean distance between estimated and groundtruth keypoints when aligning the root keypoint. That means, the distance between the estimated and the groundtruth skeleton when aligned. Since the x and y would be the same for the output and the groundtruth skeletons (because I only estimate the z coordinate) this MPJPE computing is done just using z coordinates. The formula is the following:

<img src="https://render.githubusercontent.com/render/math?math=\text{MPJPE} = \frac1T\frac1N\displaystyle\sum_{t=1}^{T}\displaystyle\sum_{i=1}^{N}\|(J_{i}^{(t)}-J_{root}^{(t)})-(\hat{J}_{i}^{(t)}-\hat{J}_{root}^{(t)})\|">

To have an idea of whether an MPJPE score is good or not, I have created the notebook `Stats.ipynb`. On that notebook I output the max, min, and mean z-distance between the keypoints within an skeleton, for every video used on the LSTMs.

*Note: For the face I use nose tip as root keypoint, in the case of the hands I use the keypoint closest to the wrist and for the body I use the middle hip keypoint.*

## Results and conclusion
It appears that in every case the LSTM learns something since the training and validation losses decrease along the epochs, and the MPJPE scores when using 5 different signers seem to be good (either using all the keypoints or just face, hands or body). 

I also noticed that if I increase the lr too much, the model tends to learn the average value of z (0. when normalized) -e.g. in "LSTM_2D_to_3D-Face.ipynb" it almost happened, since the majority of normalized predicted z were 0. on the last batch of the last epoch-. 

Now it's time to decide the next steps on this approach, and improve the results by working with the full dataset and by tuning the learning rate and the number of epochs. I could also try to use the AMSGrad variant of Adam optimizer -which is the one that VideoPose3d uses-, or add another layer to the LSTM, etc.
