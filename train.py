# CUDA_VISIBLE_DEVICES=0,1 python train.py --gpu-ids 0 1 --dataset=akshay --store-dir=20200918_akshay_trial1 --save-test-val-results --architecture=unetsar --criterion-g=l1loss
# CUDA_VISIBLE_DEVICES=2,3 python train.py --gpu-ids 0 1 --dataset=arun   --store-dir=20200918_arun_trial1   --save-test-val-results --architecture=unetsar --criterion-g=l1loss
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --gpu-ids 0 1 2 3 --dataset=arun_2   --store-dir=20200925_arun_2   --save-test-val-results --architecture=unetsar --criterion-g=l1loss
## 2020-10-01
# Studying the role of autocasting (BIGGEST DIFFERENCE - SLOWS IT DOWN), num_GPUS (ALSO SLOWS IT DOWN!!), num_workers, flattening, tensorboard metric writing...
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_2   --store-dir=20201001_speedtest_Arun   --save-test-val-results --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000
# Expt 1 - A higher batch size with no scheduling with 1e-4 learning rate - MESSED UP - REDO
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_2   --store-dir=20201001_arun_2   --save-test-val-results --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000 --batch-size=50 --adam-lr=1e-4 --lr-step-size=100000
# Expt 2 - Training on akshay's data as he did to reproduce his results
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=akshay   --store-dir=20201001_akshay   --save-test-val-results --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000
# Expt 3 - Training on arun_2 data as akshay did (his parameters) to hopefully improve upon my results from last week (by training for longer) - - MESSED UP - REDO
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_2   --store-dir=20201001_arun_2   --save-test-val-results --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000
## 2020-10-08
# Expt 1 - No scheduling with 1e-4 learning rate more epochs and testing on real data - best val of 29.12 @ step 128 and best test of 9.24 @ step 156 - 200 epochs suffice
# SUDDEN PHASE TRANSITION AT AROUND EPOCH 40... for both val and test....
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_2_realtestdata   --store-dir=20201008_arun_2_nolrscheduler    --save-test-val-results --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_2_realtestdata   --store-dir=20201008_arun_2_nolrscheduler    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - Training on arun_2_realtestdata with akshay's batchsize and learning rate scheule - best val of 33.54 @ step 149 and best test of 8.16 @ step 127
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_2_realtestdata   --store-dir=20201008_arun_2_yeslrscheduler   --save-test-val-results                                                                                    --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000  --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_2_realtestdata   --store-dir=20201008_arun_2_yeslrscheduler   --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_2_yeslrscheduler/model_epoch069.pt --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000  --epochs=1000
# Expt 3 - No scheduling with 1e-4 learning rate larger batch size and more epochs and testing on real data - best val of 31.97 @ step ~300 and best outlier test of 8.3 around epoch 60 (settles around 7.5 at ~epoch300)
# SUDDEN PHASE TRANSITION AT AROUND EPOCH 280.... for both val and test... though test while better isn't significantly better than what's already seen
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_2_realtestdata   --store-dir=20201008_arun_2_nolrscheduler_largebatch    --save-test-val-results                                                                                                --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000 --batch-size=64 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_2_realtestdata   --store-dir=20201008_arun_2_nolrscheduler_largebatch    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_2_nolrscheduler_largebatch/model_epoch135.pt   --architecture=unetsar --criterion-g=l1loss --test-batch-size=1000 --batch-size=64 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000

## 2020-10-10
# Expt 1 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) - previous best val score on 1st two sequences - best val of 36.88 @ step <200 (sim) and best test of 12.13 @ step 115 - 200 epochs suffice 
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_3_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs    --save-test-val-results                                                                                                  --architecture=unetsar_arun --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_3_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) - previous best val score on 1st two sequences - best val of 45.39 @ step 733 (sim) and best test of 10.87 @ step 28
# BETTER LOSS PERFORMANCE ON SIMULATED DATA!!! Slightly worse on real data...
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_3_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss    --save-test-val-results                                                                                                          --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_3_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt    --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - No scheduling with 1e-4 learning and testing on real data (1st two sequences) - 2D (with 3 slow time columns) - fasttime first - best val of 45.1 @ step 801 (sim) and best test of 11.94 @ step 96
# SLIGHTLY BETTER THAN EXPT 4 IN BOTH SIM AND TEST
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_2D_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst  s  --save-test-val-results                                                                                                                  --architecture=unet2d_fastfirst_3 --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_2D_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt     --architecture=unet2d_fastfirst_3 --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 4 - No scheduling with 1e-4 learning and testing on real data (1st two sequences) - 2D (with 3 slow time columns) - slowtime first - best val of 43.14 @ step 801 (sim) and best test of 10.3 @ step 45
# SLIGHT IMPROVEMENT IN VAL SNR, SLIGHT DECREASE IN TEST SNR
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_2D_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst    --save-test-val-results                                                                                                                  --architecture=unet2d_slowfirst_3 --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_2D_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt     --architecture=unet2d_slowfirst_3 --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 2020-10-23
# Sweeping 4 values of L1andFFTLoss - Expt 0 (previous day, default): (1,10), Expt1: (1, 50)  Expt2: (1, 5), Expt3:(1,1),  Expt4:(1,0.5) (Case1-4 below)
# Expt 1 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) (manually editing combo weights)
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,50    --save-test-val-results                                                                                                             --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,50    --save-test-val-results --resume --checkpoint=checkpoints/20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,50/best_model.pt    --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) (manually editing combo weights)
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,5    --save-test-val-results                                                                                                              --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,5    --save-test-val-results --resume --checkpoint=checkpoints/20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,5/best_model.pt      --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) (manually editing combo weights)
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,1    --save-test-val-results                                                                                                              --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,1    --save-test-val-results --resume --checkpoint=checkpoints/20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,1/best_model.pt      --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 4 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) (manually editing combo weights)
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,05    --save-test-val-results                                                                                                             --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs   --store-dir=20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,05    --save-test-val-results --resume --checkpoint=checkpoints/20201023_arun_realtestdata_onlyfirsttwoseqs_fftloss_1,05/best_model.pt    --architecture=unetsar_arun --criterion-g=l1andfftloss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## On 10.160.199.26
# Expt 1 - No scheduling with 1e-4 learning rate more epochs and testing on real data and training on real distributed training data
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_testdistributed             --store-dir=20201023_arun_testdistributed              --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - No scheduling with 1e-4 learning rate more epochs and testing on real data and training on real distributed training data+simulated training data
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_testdistributedandregular   --store-dir=20201023_arun_testdistributedandregular    --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## Running Akshay's network instead...
# Expt 1 - No scheduling with 1e-4 learning rate more epochs and testing on real data (1st two sequences) - previous best val score on 1st two sequences - best val of 36.88 @ step <200 (sim) and best test of 12.13 @ step 115 - 200 epochs suffice 
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_3_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs    --save-test-val-results                                                                                                  --architecture=unetsar_arun --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# TO RESUME
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_3_realtestdata_onlyfirsttwoseqs   --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs    --save-test-val-results --resume --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 2020-10-29
# Expt 1 - lr=1e-4, training on my data using Akshay's network architecture to see how it does... NOTE: While --dataset=arun_realtestdata_onlyfirsttwoseqs, I made a one time modification to it to generate 1000x1 data for Akshay's n/w
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs --store-dir=20201029_akshay_train_set_arun --save-test-val-results  --architecture=unetsar --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - STOPPED (didn't do well) - lr=1e-4, training on C-data sparse codes but testing on T-data sparse codes
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_testdistributed_Csplittrain_Tsplittest --store-dir=20201029_arun_testdistributed_Csplittrain_Tsplittest --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - lr=1e-4, training on T-data sparse codes but testing on C-data sparse codes
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_testdistributed_Tsplittrain_Csplittest --store-dir=20201029_arun_testdistributed_Tsplittrain_Csplittest --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 4 - lr=1e-4, training on C1,C2,C3,T1,T2 and testing on C4,C5,T3,T4,T5
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest --store-dir=20201029_arun_testdistributed_CTsplittrain_CTsplittest --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 replacement - lr=1e-4, training on training on C1,C2,C3,T1,T2 + simulations and testing on C4,C5,T3,T4,T5
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest_andregular --store-dir=20201029_arun_testdistributed_CTsplittrain_CTsplittest_andregular --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest_andregular --store-dir=20201029_arun_testdistributed_CTsplittrain_CTsplittest_andregular --save-test-val-results --resume --checkpoint=checkpoints/20201029_arun_testdistributed_CTsplittrain_CTsplittest_andregular/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 10.160.199.26
# Expt 0 interference - lr=1e-4, training on training on C1,C2,C3,T1,T2 + simulations and testing on C4,C5,T3,T4,T5
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_interference_testdistributed_CTsplittrain_CTsplittest_andregular --store-dir=20201029_arun_interference_testdistributed_CTsplittrain_CTsplittest_andregular --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_interference_testdistributed_CTsplittrain_CTsplittest_andregular --store-dir=20201029_arun_interference_testdistributed_CTsplittrain_CTsplittest_andregular --save-test-val-results --resume --checkpoint=checkpoints/20201029_arun_testdistributed_CTsplittrain_CTsplittest_andregular/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 10.160.199.26 - 2020-10-31
# Expt 0 - lr=1e-4, training on C2,C3,T2,T3,T4 and testing on C1,C4,C5,T1,T5
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittest --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittest --save-test-val-results --resume --checkpoint=checkpoints/20201031_arun_testdistributed_CTsplittrain_CTsplittest/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 1 - lr=1e-4, training on C2,C3,T2,T3,T4 + simulations and testing on C1,C4,C5,T1,T5
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittestandregular --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittest --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittestandregular --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittestandregular --save-test-val-results --resume --checkpoint=checkpoints/20201031_arun_testdistributed_CTsplittrain_CTsplittestandregular/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - lr=1e-4, training on C2,C3,T2,T3,T4 and testing on C1,C4,C5,T1,T5  - interference
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest_interference --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittest_interference --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittest_interference --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittest_interference --save-test-val-results --resume --checkpoint=checkpoints/20201031_arun_testdistributed_CTsplittrain_CTsplittest_interference/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - lr=1e-4, training on C2,C3,T2,T3,T4 + simulations and testing on C1,C4,C5,T1,T5 - interference
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittestandregular_interference --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittestandregular_interference --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_testdistributed_CTsplittrain_CTsplittestandregular_interference --store-dir=20201031_arun_testdistributed_CTsplittrain_CTsplittestandregular_interference --save-test-val-results --resume --checkpoint=checkpoints/20201031_arun_testdistributed_CTsplittrain_CTsplittestandregular_interference/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - lr=1e-4, training on sim data
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_interference --store-dir=20201031_arun_interference --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_interference --store-dir=20201031_arun_interference --save-test-val-results --resume --checkpoint=checkpoints/20201031_arun_interference/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 10.160.199.26 - 2020-11-01
# Expt 2 - lr=1e-4, training on forward looking --exact-- data
# CUDA_VISIBLE_DEVICES=2 python train.py --gspu-ids 0 --dataset=arun_fwd_exact --store-dir=20201101_arun_fwd_exact --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_fwd_exact --store-dir=20201101_arun_fwd_exact --save-test-val-results --resume --checkpoint=checkpoints/20201101_arun_fwd_exact/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - lr=1e-4, training on forward looking --generated-- data
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_fwd_generated --store-dir=20201031_arun_fwd_generated --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_fwd_generated --store-dir=20201031_arun_fwd_generated --save-test-val-results --resume --checkpoint=checkpoints/20201031_arun_fwd_generated/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 0 - lr=1e-4, training on forward looking --exact-- data
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_fwd_exact --store-dir=20201101_arun_fwd_exact_fftloss --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_fwd_exact --store-dir=20201101_arun_fwd_exact_fftloss --save-test-val-results --resume --checkpoint=checkpoints/20201101_arun_fwd_exact_fftloss/best_model.pt  --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 1 - lr=1e-4, training on forward looking --generated-- data
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_fwd_generated --store-dir=20201101_arun_fwd_generated_fftloss --save-test-val-results  --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_fwd_generated --store-dir=20201101_arun_fwd_generated_fftloss --save-test-val-results --resume --checkpoint=checkpoints/20201101_arun_fwd_generated_fftloss/best_model.pt  --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 10.160.199.26 - 2020-11-02
# Expt 0 - lr=1e-4, training on interference data
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_interference_testonfirsttwoseqs --store-dir=20201102_arun_interference_testonfirsttwoseqs --save-test-val-results                                                                                                --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_interference_testonfirsttwoseqs --store-dir=20201102_arun_interference_testonfirsttwoseqs --save-test-val-results --resume --checkpoint=checkpoints/20201102_arun_interference_testonfirsttwoseqs/best_model.pt  --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 1 - lr=1e-4, training on data with multiple missing frequencues
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates --store-dir=20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates --save-test-val-results                                                                                                                     --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates --store-dir=20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates --save-test-val-results --resume --checkpoint=checkpoints/20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - lr=1e-4, training on data with multiple missing frequencues - random gaps
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps --store-dir=20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps --save-test-val-results                                                                                                                                 --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps --store-dir=20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps --save-test-val-results --resume --checkpoint=checkpoints/20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - lr=1e-4, training on data with multiple missing frequencues - random gaps
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_generative_modeled_realtestdata_onlyfirsttwoseqs --store-dir=20201102_arun_generative_modeled_realtestdata_onlyfirsttwoseqs --save-test-val-results                                                                                                                   --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_generative_modeled_realtestdata_onlyfirsttwoseqs --store-dir=20201102_arun_generative_modeled_realtestdata_onlyfirsttwoseqs --save-test-val-results --resume --checkpoint=checkpoints/20201102_arun_generative_modeled_realtestdata_onlyfirsttwoseqs/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 4 - lr=1e-4, training on data with multiple missing frequencues - random gaps
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_generative_modeled_realtestdata_onlyfirsttwoseqs --store-dir=20201102_arun_generative_modeled_realtestdata_onlyfirsttwoseqs_withfftloss --save-test-val-results                                                                                                                               --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_generative_modeled_realtestdata_onlyfirsttwoseqs --store-dir=20201102_arun_generative_modeled_realtestdata_onlyfirsttwoseqs_withfftloss --save-test-val-results --resume --checkpoint=checkpoints/20201102_arun_generative_modeled_realtestdata_onlyfirsttwoseqs_withfftloss/best_model.pt    --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 10.160.199.26 - 2020-11-03
# Expt 0 - lr=1e-4, training on extended simulations + generative model data
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs --save-test-val-results                                                                                                                 --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs --save-test-val-results --resume --checkpoint=checkpoints/20201103_arun_extended_and_generative_testononlyfirsttwoseqs/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 1 - lr=1e-4, training on extended simulations + generative model data + fftloss
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_fftloss --save-test-val-results                                                                                                                         --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_fftloss --save-test-val-results --resume --checkpoint=checkpoints/20201103_arun_extended_and_generative_testononlyfirsttwoseqs_fftloss/best_model.pt    --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 2 - lr=1e-4, training on extended simulations + generative model data + INTERFERENCE
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_interference --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_interference --save-test-val-results                                                                                                                               --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=2 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_interference --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_interference --save-test-val-results --resume --checkpoint=checkpoints/20201103_arun_extended_and_generative_testononlyfirsttwoseqs_interference/best_model.pt     --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 3 - lr=1e-4, training on extended simulations + generative model data + RANDOMGAPS (50% missing)
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps --save-test-val-results                                                                                                                               --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=3 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps --save-test-val-results --resume --checkpoint=checkpoints/20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps/best_model.pt       --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
## 10.160.199.158 - 2020-11-03
# Expt 0 - lr=1e-4, training on extended simulations + generative model data + BLOCKGAPS_ALL - ended up running on GPU#1 of .26
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_allrates --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_allrates --save-test-val-results                                                                                                                                    --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_allrates --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_allrates --save-test-val-results --resume --checkpoint=checkpoints/20201103_arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_allrates/best_model.pt    --architecture=unetsar_arun --criterion-g=l1loss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Expt 1 - lr=1e-4, training on extended simulations + generative model data + RANDOMGAPS_ALL - ended up running on GPU#0 of .26
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_allrates --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_allrates --save-test-val-results                                                                                                                                     --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000
# Resume
# CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 0 --dataset=arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_allrates --store-dir=20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_allrates --save-test-val-results --resume --checkpoint=checkpoints/20201103_arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_allrates/best_model.pt    --architecture=unetsar_arun --criterion-g=l1andfftloss --batch-size=10 --test-batch-size=1000 --adam-lr=1e-4 --lr-step-size=100000 --epochs=1000

import time
import numpy as np
import random
import logging
import argparse, pathlib
import os, sys, glob
import time
import pdb
import shutil
import sklearn

# PyTorch DNN imports
import torch
import torch.nn
import torch.optim
import torchvision 

# Torch Mixed-Precision Training Imports
from torch.cuda.amp import GradScaler, autocast

# U-Net related imports
# from unet import GeneratorUnet1_1, GeneratorUnet1_1_FAIR, UNetSeparable_64_uros, UNetSeparable_64_uros_small, UNetSeparable_64_uros_small_5, UNetSeparable_64, UNetSeparable_16, visualize_neurons, UNet1Dk5s2, UNet1Dk5s2_siren, UNet1Dk15s4
from models import UNetSAR, UNetSAR_Arun, UNet2DSAR_fastfirst_3, UNet2DSAR_slowfirst_3

# Loss Function Imports
# from utils import DiceCoeffLoss, RMSELoss, LSDLoss
from utils import L1andFFTLoss

# Tensorboard import
from torch.utils.tensorboard import SummaryWriter

# Dataset imports
# from utils import retrieve_dataset_filenames
from utils.dataset_deets import dataset_deets

# Dataloader imports
from torch.utils.data import DataLoader
from torchvision import transforms
# from utils import MyDataset, MyDataset_Kaz_Training_Raw, ApplySTFT, ApplyPacketLoss, ApplyTelecomDistortions, RandomSubsample, TFGapFillingMaskEstimation, ConcatMaskToInput, ToTensor
# from utils import PoseInterpolatorFCDataset, ToTensor, PoseSubsampler

# Validation/Testing data related inputs
# from utils import eval_net
from utils import eval_net, write_imgs

# # Import NVIDIA AMP for mixed precision arithmetic
# try:
#     # sys.path.append('/home/t-arnair/Programs/apex')
#     from apex import amp
#     APEX_AVAILABLE = True
#     DROP_LAST = True # Need to drop the last batch or an out-of-memory error occurs
#     # OPT_LEVEL = "O2"
#     OPT_LEVEL = "O1"
# except ModuleNotFoundError:
#     APEX_AVAILABLE = False
#     DROP_LAST = False
# # TO OVERRIDE AND JUST IGNORE APEX if necessary
# # APEX_AVAILABLE = False
# # DROP_LAST = False

# Set logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: To set GPU Visibilisty from inside the code
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# -------------------------------------------------------------------------------------------------
# # Section I - train, evaluate, and visualize functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Training function for one epoch
def train_epoch(params, epoch, g_net, criterion_g, train_loader, optimizer_G, scaler, scheduler, writer):
    print('-'*80)
    print('Set the neural network to training mode...')
    print('-'*80)
    
    # Need to do this to ensure batchnorm and dropout layers exhibit training behavior
    g_net.train() 

    # Initialize the loss for this epoch to 0
    g_epoch_loss = 0.
    # avg_loss = 0. - FAIR code


    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader) # len(train_loader) is the number of batches to process in one training epoch

    tot_epochs = params['epochs']

    for iterIdx, sample in enumerate(train_loader):
        # Preprocess the data (if required)
        # input_data = sample['input_data']
        # target_output = sample['target_output']
        target_output, input_data = sample

        # Copy the data to GPU
        input_data = input_data.to(params['device'])
        target_output = target_output.to(params['device'])
        # target_output_flat = target_output.view(-1) # Flatten (i.e.) vectorize the data

        # Zero the parameter gradients
        optimizer_G.zero_grad()

        # Pass the minibatch through the network to get predictions WITH AUTOCASTING
        with autocast(enabled=False): # Had to disable it since it made performance workse surprisingly
            preds = g_net(input_data)            
            # preds = torch.sigmoid(preds) # NOTE: Don't need the sigmoid... just removing it in favor of a plain conv like Dung did...
            # preds_flat = preds.view(-1)

            # Calculate the loss for the batch
            # if params['criterion_g']=='lsdloss': # Here, can't use flattened data
            #     loss = criterion_g(preds, target_output)
            # else: # Otherwise, needs flattened data
            #     loss = criterion_g(preds_flat, target_output_flat)
            if params['criterion_g'] == 'l1andfftloss':
                loss = criterion_g(input_data, preds, target_output)
            else:                
                loss = criterion_g(preds, target_output)
                
        # # Compute the gradients and take a step
        # if APEX_AVAILABLE:
        #     with amp.scale_loss(loss, optimizer_G) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        scaler.scale(loss).backward()

        # optimizer_G.step()
        scaler.step(optimizer_G)

        # Updates the scale for next iteration.
        scaler.update()

        
        g_epoch_loss += loss.item() # Note: calling .item() creates a a copy which is stored in g_epoch_loss... so modifying the latter doesn't modify the former. Verified!
        # avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item() - FAIR - dunno what is going on here...
        writer.add_scalar(f"Train/TrainLoss", loss.item(), global_step + iterIdx)

        if iterIdx % params['log_interval'] == 0:
            # logging.info(
            #     f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            #     f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            #     f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
            #     f'Time = {time.perf_counter() - start_iter:.4f}s',
            # )
            logging.info(
                f'Epoch = [{epoch+1:3d}/{ tot_epochs:3d}] '
                f'Iter = [{iterIdx:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {g_epoch_loss/(iterIdx+1):.4g} '
                f'MiniBatch Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    # scheduler.step(epoch) # Wasn't sure if it should be epoch or epoch+1, so just doing step() - step has well defined behavior
    scheduler.step()

    # ReDefine avg_loss to what makes more sense to me instead of the FAIR code
    avg_loss = g_epoch_loss/(iterIdx+1)

    writer.add_scalar(f"Train/TrainLoss_EpochAvg", avg_loss, epoch)

    return avg_loss, time.perf_counter() - start_epoch

# -------------------------------------------------------------------------------------------------
## Evaluation function for val/test data
def evaluate(params, epoch, g_net, criterion_g, data_loader, writer, data_string):
    
    print('-'*80)
    print('Set the neural network to testing mode...')
    print('-'*80)
    # Set the network to eval mode to freeze BatchNorm weights
    g_net.eval()
    start = time.perf_counter()
    _l1loss, _mseloss, _smoothl1loss, _snrloss, _snrloss_donothing, _snrloss_gain, _loss  = eval_net(g_net, criterion_g, params, data_string, data_loader)

    print(f"{data_string} L1 Loss (Lower is better): {_l1loss}")
    print(f"{data_string} Smooth L1 Loss (Lower is better): {_smoothl1loss}")
    print(f"{data_string} MSE Loss (Lower is better): {_mseloss}")
    print(f"{data_string} SNR (Higher is better): {_snrloss}")
    print(f"{data_string} SNRDoNothing (Higher is better): {_snrloss_donothing}")
    print(f"{data_string} SNR Gain (Higher is better): {_snrloss_gain}")
    print(f"{data_string} {params['criterion_g']} Loss (Lower is better): {_loss}")

    writer.add_scalar(f"Loss/{data_string}_L1",  _l1loss, epoch)
    writer.add_scalar(f"Loss/{data_string}_SmoothL1",  _smoothl1loss, epoch)
    writer.add_scalar(f"Loss/{data_string}_MSE",  _mseloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_SNR",  _snrloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_SNRDoNothing",  _snrloss_donothing, epoch)
    writer.add_scalar(f"Loss/{data_string}_SNRGain",  _snrloss_gain, epoch)
    # writer.add_scalar(f"Loss/{data_string}_psnr", _PSNR, epoch)
    writer.add_scalar(f"Loss/{data_string}_loss", _loss, epoch)

    mean_points = _snrloss
    # mean_points = -1*_loss
    mean_loss = _loss
    # return np.mean(losses), time.perf_counter() - start
    return mean_points, mean_loss, time.perf_counter() - start

# -------------------------------------------------------------------------------------------------
## Visualization function for network output -- track it using tensorboard
def visualize(params, epoch, g_net, data_loader, writer):
    def save_image(image, tag, nrow):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=nrow, pad_value=1)
        writer.add_image(tag, grid, epoch)

    g_net.eval()
    with torch.no_grad():
        for iterIdx, sample in enumerate(data_loader):

            # input, target, mean, std, norm = data
            # input = input.unsqueeze(1).to(args.device)
            # target = target.unsqueeze(1).to(args.device)
            # output = model(input)

            # input_data = sample['input_data'].to(params['device']) 
            # target_output = sample['target_output'].to(params['device']) 

            target_output, input_data = sample
            input_data = input_data.to(params['device'])
            target_output = target_output.to(params['device'])
            
            preds = g_net(input_data)
            # preds = torch.sigmoid(preds) # No sigmoid!

            save_image(target_output, 'Images/Target', nrow=1)
            save_image(preds, 'Images/Reconstruction', nrow=1)
            save_image(torch.abs(target_output - preds), 'Images/Error', nrow=1)
            break

# -------------------------------------------------------------------------------------------------
# # Section II - Dataset definition and dataloader creation
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Create dataset objects to load the data
def create_datasets(params):
    train_data, val_data, test_data = dataset_deets(params)
    return train_data, val_data, test_data



# -------------------------------------------------------------------------------------------------
## Create dataloaders to load data from my dataset objects
def create_data_loaders(params):
    # train_data, val_data, test_data, dataset_deets, req_filenames_dict = create_datasets(params)
    train_data, val_data, test_data = create_datasets(params)

    display_data_skipfactor = 20
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)] 
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 64)] 
    display_data = [test_data[i] for i in range(0, len(test_data), len(test_data) // display_data_skipfactor)] 
    
    train_loader    = torch.utils.data.DataLoader(train_data,   batch_size=params['batch_size'], shuffle=True,      num_workers=len(params['gpu_ids']))
    val_loader      = torch.utils.data.DataLoader(val_data,     batch_size=params['test_batch_size'], shuffle=False,     num_workers=len(params['gpu_ids']))
    test_loader     = torch.utils.data.DataLoader(test_data,    batch_size=params['test_batch_size'], shuffle=False,     num_workers=len(params['gpu_ids']))
    display_loader  = torch.utils.data.DataLoader(display_data, batch_size=display_data_skipfactor, shuffle=False,  num_workers=len(params['gpu_ids']))
    # train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'],
    #                             shuffle=True, num_workers= 0*len(params['gpu_ids']), pin_memory=True, drop_last=DROP_LAST) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    # val_loader   = DataLoader(dataset=val_data, batch_size=params['test_batch_size'],
    #                             num_workers= 0*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    # test_loader  = DataLoader(dataset=test_data, batch_size=params['test_batch_size'],
    #                             num_workers= 0*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    return train_loader, val_loader, test_loader, display_loader


# -------------------------------------------------------------------------------------------------
# # Section III - Argument parser and argument dictionary creation
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Argument parser
def parse_args(args):
    parser = argparse.ArgumentParser(description='PyTorch RADAR interference Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## Parameters to update every run (or thereabouts)
    # IMPORTANT: usage --gpu-ids 0 1 2 3
    parser.add_argument('--gpu-ids', nargs='+', default=[0], type=int, 
                        help='List of GPU-IDs to run code on')
    # IMPORTANT: usage --dataset=DNS-challenge-synthetic-test
    parser.add_argument('--dataset', required=True, type=str,
                        help='<predefined dataset name>|<path to dataset>')
    # IMPORTANT: usage --store-dir=20190102_20200102_Sim1
    parser.add_argument('--store-dir', required=True, type=pathlib.Path,
                        help='Name of output directory')
    parser.add_argument('--save-test-val-results', action='store_true', default=False,
                        help='Whether to save val and test outputs in files')
    ## Parameters to update less often
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')    
    parser.add_argument('--checkpoint', type=str, default=False,
                        help='Path to an existing checkpoint. Used along with "--resume"')    
    parser.add_argument('--prng-seed', type=int, default=1337, metavar='S',
                        help='Seed for all the pseudo-random number generators')
    parser.add_argument('--architecture', choices=['unetsar', 'unetsar_arun', 'unet2d_fastfirst_3', 'unet2d_slowfirst_3'], default='unetsar_arun', type=str,
                        help='unetsar|unetsar_arun|unet2d_fastfirst_3|unet2d_slowfirst_3|...')
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help='Flag to prevent paralellization of the model across the GPUs')
    parser.add_argument('--in-chans', default=1, type=int, metavar='IC',
                        help='Number of input channels') # TODO - adapt this to set edge_sample_length and sample_gap
    # parser.add_argument('--normalization', choices=['none','batchnorm','instancenorm'], default='batchnorm', type=str,
    #                     help='none|batchnorm|instancenorm') # TODO - adapt this according to Oscar's code...
    parser.add_argument('--criterion-g', choices=['l1loss', 'smoothl1loss', 'mseloss', 'l1andfftloss'], required=True,
                        help='l1loss|smoothl1loss|mseloss|l1andfftloss')
    # IMPORTANT: usage --epochs=80 or --epochs 80 (both work)
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='Input mini-batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N_T',
                        help='Input mini-batch size for testing (default: 1000)') # need it to be 1, otherwise different sequences have different numbers of frames...
    parser.add_argument('--trunc-data-flag', action='store_true', default=False, # TODO - this isn't implemented yet - I should do it
                        help='Work with truncated dataset')
    parser.add_argument('--adam-lr', type=float, default=0.0005, metavar='LR',
                        help='Adam learning rate (default: 5e-4)')
    parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='AB1',
                        help='Adam beta 1 term (default: 0.9)')
    parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='AB2',
                        help='Adam beta 2 term (default: 0.999)')
    parser.add_argument('--lr-step-size', type=int, default=20,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                        help='Multiplicative factor of learning rate decay')
    # parser.add_argument('--flip-training', action='store_true', default=False,
    #                     help='Whether to flip the training data')
    # parser.add_argument('--test-rows', type=int, default=128, metavar='R',
    #                     help='Number of rows in the test data inputs (default: 128)')
    # parser.add_argument('--test-cols', type=int, default=128, metavar='C',
    #                     help='Number of cols in the test data inputs (default: 128)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='How many mini-batches to wait before logging training status')
    parser.add_argument('--teststats-save-interval', type=int, default=1, metavar='TSI',
                        help='How many epochs to wait before testing on val/test data and saving stats to tensorboard')
    parser.add_argument('--checkpoint-interval', type=int, default=20, metavar='CI',
                        help='How many epochs to wait before saving a model file')
    return parser.parse_args(args)

# -------------------------------------------------------------------------------------------------
## Convert parsed arguments to a dict for more easy manipulation
def create_params_dict(parsed_args, device):
    params = {}
    params['gpu_ids']           = parsed_args.gpu_ids                     # ids of GPUs being used
    params['dataset']           = parsed_args.dataset                     # Which dataset to processs
    params['checkpoints_dir']   = os.path.join("checkpoints", parsed_args.store_dir)   # Directory in which to store the model checkpoints and training details as .pt files
    params['results_dir']       = os.path.join("results", parsed_args.store_dir)       # Directory in which to store the outputs
    params['save_test_val_results'] = parsed_args.save_test_val_results   # Whether to save val and test outputs
    params['resume']            = parsed_args.resume                      # Flag set to true if training is to be resumed from provided checkpoint
    params['checkpoint']        = parsed_args.checkpoint                  # Where to load the model from if it is being loaded; false otherwise
    params['architecture']      = parsed_args.architecture                # Set to 'og', 'fair', 'unetseparable_uros', 'unetseparable', 'unet1dk5s2', or 'unet1dk15s4' - determines the neural network architecture implementation
    params['no_parallel']       = parsed_args.no_parallel                 # Flag to prevent model paralellization across GPUs
    params['in_chans']          = parsed_args.in_chans                    # Number of input channels - defaults to 1
    # params['normalization']     = parsed_args.normalization               # Set to 'none', 'batchnorm', or 'instancenorm'
    params['criterion_g']       = parsed_args.criterion_g                 # DNN loss function
    params['epochs']            = parsed_args.epochs                      # Total number of training epochs i.e. complete passes of the training data
    params['batch_size']        = parsed_args.batch_size                  # Number of training files in one mini-batch
    params['test_batch_size']   = parsed_args.test_batch_size             # Number of testing files in one mini-batch - can be much larger since gradient information isn't required
    params['trunc_data_flag']   = parsed_args.trunc_data_flag             # Flag on whether to truncate the dataset
    params['adam_lr']           = parsed_args.adam_lr                     # Learning rate for the Adam Optimzer
    params['adam_beta1']        = parsed_args.adam_beta1                  # Adam beta 1 term
    params['adam_beta2']        = parsed_args.adam_beta2                  # Adam beta 2 term
    params['lr_step_size']      = parsed_args.lr_step_size                # Period of learning rate decay - after this number of epochs, the lr will decrease by the multiplicative factor of lr_gamma
    params['lr_gamma']          = parsed_args.lr_gamma                    # Multiplicative factor of learning rate decay. Default: 0.1.    
    # params['to_flip']           = int(parsed_args.flip_training)          # Whether to augment the training data through flipping it laterally
    # params['test_rows']         = parsed_args.test_rows                   # Number of rows in the test images
    # params['test_cols']         = parsed_args.test_cols                   # Number of columns in the test images
    params['log_interval']              = parsed_args.log_interval              # How often to print loss information for a minibatch
    params['teststats_save_interval']   = parsed_args.teststats_save_interval   # How often to run test data and save test stats
    params['checkpoint_interval']       = parsed_args.checkpoint_interval       # How often to save a snapshot of the model
    params['device']            = device                                  # Device to run the code on

    # Create a directory to store checkpoints if it doesn't already exist
    os.makedirs(params['checkpoints_dir'], exist_ok=True)

    return params


# -------------------------------------------------------------------------------------------------
# # Section IV - Smaller Utility Functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Seed all the random number generators
def seed_prng(parsed_args, device):
    # Do the below to ensure reproduceability - from the last comment at
    # https://discuss.pytorch.org/t/random-seed-initialization/7854/18
    # NOTE: I didn't do the num_workers thing they suggested, but reproducibility
    # was obtained without it
    # NOTE: According to FAIR fastMRI code, the three below lines will suffice for reproducibility... it doesn't set the CUDA seeds.
    np.random.seed(parsed_args.prng_seed)
    random.seed(parsed_args.prng_seed)
    torch.manual_seed(parsed_args.prng_seed)

    # if you are using GPU # This might not be necessary, as per https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
    if 'cuda' in device.type:
        print(f"Using the following GPUs: {parsed_args.gpu_ids}")
        torch.cuda.manual_seed(parsed_args.prng_seed)
        torch.cuda.manual_seed_all(parsed_args.prng_seed) 
        torch.backends.cudnn.enabled = True # This was originally false. Changing it to true still seems to work.
        torch.backends.cudnn.benchmark = False # Sometimes need to change this to false or else you get a CUDNN_STATUS_INTERNAL_ERROR
        # torch.backends.cudnn.benchmark = True # Changing this to true from false 1) Speeds up the code a bit - 160s instead of 180s 2) Mostly preserves deterministic behavior (ran it thrice, runs 2 and 3 were identical, 1 was slightly different)
        torch.backends.cudnn.deterministic = True # Changing this to false for my use case didn't make a difference in speed

# -------------------------------------------------------------------------------------------------
## Build the DNN model
def build_model(parsed_args, device):
    # Initialize the neural network model
    if parsed_args.architecture == 'unetsar':        
        g_net = UNetSAR()
    elif parsed_args.architecture == 'unetsar_arun':
        g_net = UNetSAR_Arun()
    elif parsed_args.architecture == 'unet2d_fastfirst_3':
        g_net = UNet2DSAR_fastfirst_3()
    elif parsed_args.architecture == 'unet2d_slowfirst_3':
        g_net = UNet2DSAR_slowfirst_3()
    else:
        print('Unacceptable input arguments when building the network')
        sys.exit(0)
    
    # Move the model to the GPUs
    g_net.to(device)
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    return g_net, scaler

# -------------------------------------------------------------------------------------------------
## Load the DNN model from disk
def load_model(parsed_args, device):
    checkpoint = torch.load(parsed_args.checkpoint)
    print(f"Generator Model loaded from {parsed_args.checkpoint}")
    
    parsed_args = checkpoint['parsed_args']
    g_net, scaler = build_model(parsed_args, device)
    optimizer_G, scheduler = build_optim(parsed_args, g_net)    

    checkpoint['g_net'] = {key.replace('module.', ''):value for key, value in checkpoint['g_net'].items()} # Need to ``un-parallelize'' the dict before loading it... - because of the amp change
    # While amp is no longer used, it's still useful code to have to allow de- and re-parallelization
    g_net.load_state_dict(checkpoint['g_net'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    scaler.load_state_dict(checkpoint['scaler'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint, g_net, optimizer_G, scaler, scheduler

# -------------------------------------------------------------------------------------------------
## Build the optimizer
def build_optim(parsed_args, g_net):
    # Specify the optimizer's parameters - this is the optimzation algorithm that trains the  neural network
    optimizer_G = torch.optim.Adam(g_net.parameters(), lr=parsed_args.adam_lr, betas=(parsed_args.adam_beta1, parsed_args.adam_beta2), weight_decay=0)

    # Initialize a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, parsed_args.lr_step_size, parsed_args.lr_gamma)
    # NOTE: There exist other interesting schedulers like ReduceLROnPlateau that are worth checking out

    return optimizer_G, scheduler

# -------------------------------------------------------------------------------------------------
## Initialize the loss criteria
def initialize_loss_criterion(params):
    # Options for possible loss functions
    # loss_dict = {'dscloss':DiceCoeffLoss(), 'maeloss': torch.nn.L1Loss(), 'mseloss': torch.nn.MSELoss(), 'rmseloss': RMSELoss(), 'bceloss': torch.nn.BCELoss()}
    loss_dict = {'l1loss': torch.nn.L1Loss(), 'smoothl1loss': torch.nn.SmoothL1Loss(), 'mseloss': torch.nn.MSELoss(), 'l1andfftloss': L1andFFTLoss}
    # NOTE: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    # So use it in general when possible...
    
    # Set loss functionsn to use for training DNN
    criterion_g = loss_dict[params['criterion_g']]

    # Moving it to the GPU - not really required except for stateful losses as stated in https://discuss.pytorch.org/t/move-the-loss-function-to-gpu/20060
    # criterion_g.to(params['device'])

    return criterion_g

# -------------------------------------------------------------------------------------------------
## Save the model to diisk
def save_model(parsed_args, params, epoch, g_net, optimizer_G, scaler, scheduler, val_points, best_val_points, is_new_best):
    # NOTE: Save model every 10 epochs, otherwise it's excessive
    checkpoint= {
                    'epoch': epoch,
                    'parsed_args': parsed_args,
                    'g_net': g_net.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_points': val_points,
                    'best_val_points': best_val_points,
                    'checkpoints_dir': params['checkpoints_dir']
                }
    # torch.save(net.cpu(),os.path.join(conf.save_dir, 'UNet_%s_%s_epoch%d_maxsparse%d.pkl'%(conf.train_regime, conf.criterion, epoch+1, conf.max_sparse))) # To keep in mind what Akshay judged important to keep track of
    if epoch % params['checkpoint_interval'] == 0:
        torch.save(checkpoint,
                # f = exp_dir / 'model_epoch{0:0>3d}.pt'.format(epoch)
                f = os.path.join(params['checkpoints_dir'], 'model_epoch{0:0>3d}.pt'.format(epoch))
            )
    if is_new_best:
        # This is the old code when every single epoch was being saved. Here, we can just copy it.
        # # shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        # shutil.copyfile(os.path.join(params['checkpoints_dir'], 'model_epoch{0:0>3d}.pt'.format(epoch)), 
        #    os.path.join(params['checkpoints_dir'], 'best_model.pt'.format(epoch)))
        # This is the new code when it's not guaranteed to have been saved, so resave it.
        torch.save(checkpoint,
            # f = exp_dir / 'model_epoch{0:0>3d}.pt'.format(epoch)
            f = os.path.join(params['checkpoints_dir'], 'best_model.pt')
        )

# -------------------------------------------------------------------------------------------------
# # Section V - The Main driver function
# -------------------------------------------------------------------------------------------------

def main(args):
    print('-'*80)
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print('-'*80)

    print('-'*80)
    print('Parsing arguments...')
    print('-'*80)
    parsed_args = parse_args(args)

    # Specify the device to run the code on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('-'*80)
    print('Seeding the pseudo-random number generators...')
    print('-'*80)    
    seed_prng(parsed_args, device) 

    print('-'*80)
    print('Initializing neural network model...')
    print('-'*80)

    if parsed_args.resume:
        new_num_epochs = parsed_args.epochs
        checkpoint, g_net, optimizer_G, scaler, scheduler = load_model(parsed_args, device)
        parsed_args = checkpoint['parsed_args']
        parsed_args.epochs = new_num_epochs # Keep this as the new value
        parsed_args.resume = True # Don't want this overwritten...
        parsed_args.checkpoint_interval = 20 # TODO
        best_val_points = checkpoint['best_val_points']
        start_epoch = checkpoint['epoch']+1 # NOTE: Need the +1 since we don't want to run the saved epoch again
        del checkpoint
    else:
        g_net, scaler = build_model(parsed_args, device)
        optimizer_G, scheduler = build_optim(parsed_args, g_net)
        # best_val_loss = 1e9
        best_val_points = -20 # NOTE: if using PSNR/PESQ, want it set low
        start_epoch = 0

    logging.info(parsed_args)
    logging.info(g_net)
    
    # Create a dict out of the parsed_args
    params = create_params_dict(parsed_args, device)

    train_loader, val_loader, test_loader, display_loader = create_data_loaders(params)

    criterion_g = initialize_loss_criterion(params)

    # args.exp_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir = args.exp_dir / 'summary') # Cool way to use '/' to combine two dirs
    writer = SummaryWriter(log_dir = os.path.join(params['checkpoints_dir'], 'summary'))

    pytorch_total_params = sum(p.numel() for p in g_net.parameters() if p.requires_grad)

    try:
        print(f'''
        Starting training:
            Epochs: {params['epochs']}
            Batch size: {params['batch_size']}
            Dataset: {params['dataset']}
            Optimization Algorithm: 'Adam'
            Learning rate: {params['adam_lr']}
            Checkpoints: {str(params['checkpoints_dir'])}
            Device: {str(params['device'])}
            Trainable Nw Params: {pytorch_total_params}
        ''')

        tot_epochs = params['epochs']

        if not params['no_parallel']:
            # Set it to parallelize across visible GPUs - this needs to be after the call to amp.initialize
            g_net = torch.nn.DataParallel(g_net)
        pdb.set_trace()
        for epoch in range(start_epoch, params['epochs']):

            # Do the train step
            tic = time.perf_counter()
            train_loss, train_time = train_epoch(params, epoch, g_net, criterion_g, train_loader, optimizer_G, scaler, scheduler, writer)
            toc = time.perf_counter()
            print(f'Epoch {epoch} took {toc - tic:0.4f} seconds')

            if (epoch % params['teststats_save_interval'] == 0) or (epoch % params['checkpoint_interval'] == 0):
                # Do the validation/test step
                # val_points, val_loss, val_time     = evaluate(params, dataset_deets, epoch, g_net, criterion_g, val_loader,  writer, 'val',  req_filenames_dict['val'])
                # test_points, test_loss, test_time  = evaluate(params, dataset_deets, epoch, g_net, criterion_g, test_loader, writer, 'test', req_filenames_dict['test'])
                val_points, val_loss, val_time     = evaluate(params, epoch, g_net, criterion_g, val_loader,  writer, 'val')
                test_points, test_loss, test_time  = evaluate(params, epoch, g_net, criterion_g, test_loader, writer, 'test')
                # visualize_neurons(params, epoch, g_net, display_loader, writer)
                # visualize(params, epoch, g_net, display_loader, writer)
                # is_new_best = val_loss < best_val_loss  # NOTE: If using PSNR/PESQ, we actually want the highest loss lol
                temp = val_points # TODO: change this back to validation once your simulator is good enough...
                val_points = test_points # TODO: change this back to validation once your simulator is good enough...
                test_points = temp # TODO: change this back to validation once your simulator is good enough...
                is_new_best = val_points > best_val_points  # NOTE: If using PSNR/PESQ, we actually want the highest loss lol                
                if is_new_best:
                    write_imgs(g_net, params, 'val', val_loader)
                    write_imgs(g_net, params, 'test', test_loader)
                best_val_points = max(best_val_points, val_points)
                save_model(parsed_args, params, epoch, g_net, optimizer_G, scaler, scheduler, val_points, best_val_points, is_new_best)
                logging.info(
                    f'Epoch = [{epoch+1:4d}/{tot_epochs:4d}] TrainLoss = {train_loss:.4g} '                    
                    f'ValPoints = {val_points:.4g} TestPoints = {test_points:.4g} '
                    f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s TestTime = {test_time:.4f}s',
                )
        # Write network graph to file
        # NOTE: 1) Doesn't work with dataparallel 2) Need to use the FAIR UNet code
        # sample = next(iter(display_loader))
        # writer.add_graph(g_net, sample['input_data'].to(device))                

        writer.close()
    
        # Free up memory used
        del g_net
        torch.cuda.empty_cache()        

    except KeyboardInterrupt:
        print('Code Interrupted!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)            

if __name__ == '__main__':
    main(sys.argv[1:])

# Include MS-SSIM
# tensorboard --samples_per_plugin images=100 --bind_all --port=32000
