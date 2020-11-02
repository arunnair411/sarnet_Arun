# CUDA_VISIBLE_DEVICES=2 python test.py --gpu-ids 0 --dataset=real --data-split=test   --store-dir=20200918_akshay_trial1_real   --save-test-val-results --checkpoint=checkpoints/20200918_akshay_trial1/model_epoch100.pt --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=real --data-split=test   --store-dir=20200918_arun_trial1_real     --save-test-val-results --checkpoint=checkpoints/20200918_arun_trial1/model_epoch100.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=2 python test.py --gpu-ids 0 --dataset=arun --data-split=test   --store-dir=20200918_akshay_trial1_arun   --save-test-val-results --checkpoint=checkpoints/20200918_akshay_trial1/model_epoch100.pt --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=akshay --data-split=test --store-dir=20200918_arun_trial1_akshay   --save-test-val-results --checkpoint=checkpoints/20200918_arun_trial1/model_epoch100.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10

# Testing on inverted waveforms
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun_inverted --data-split=test   --store-dir=20200918_arun_trial1_aruninverted   --save-test-val-results --checkpoint=checkpoints/20200918_arun_trial1/model_epoch100.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun         --data-split=test --store-dir=20200918_arun_trial1_arunnotinverted   --save-test-val-results --checkpoint=checkpoints/20200918_arun_trial1/model_epoch100.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10

# 2020-09-25
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun_2         --data-split=test --store-dir=20200925_arun_2_simtest   --save-test-val-results --checkpoint=checkpoints/20200925_arun_2/model_epoch020.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=real           --data-split=test --store-dir=20200925_arun_2_real      --save-test-val-results --checkpoint=checkpoints/20200925_arun_2/model_epoch020.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=real_C1        --data-split=test --store-dir=20200925_arun_2_real_C1   --save-test-val-results --checkpoint=checkpoints/20200925_arun_2/model_epoch020.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=real_C2        --data-split=test --store-dir=20200925_arun_2_real_C2   --save-test-val-results --checkpoint=checkpoints/20200925_arun_2/model_epoch020.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=10

# 2020-10-02
# Expt 1/3 - Basically the best model at the end of my mess up.... still a good model though (32dB gain on sim data)
# CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=arun_2         --data-split=test --store-dir=20201001_arun_2_messedup_simtest   --save-test-val-results --checkpoint=checkpoints/20201001_arun_2/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=1000
# CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=real           --data-split=test --store-dir=20201001_arun_2_messedup_real      --save-test-val-results --checkpoint=checkpoints/20201001_arun_2/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=real_C1        --data-split=test --store-dir=20201001_arun_2_messedup_real_C1                           --checkpoint=checkpoints/20201001_arun_2/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=real_C2        --data-split=test --store-dir=20201001_arun_2_messedup_real_C2                           --checkpoint=checkpoints/20201001_arun_2/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=akshay         --data-split=test --store-dir=20201001_arun_2_messedup_akshay    --save-test-val-results --checkpoint=checkpoints/20201001_arun_2/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=1000
# CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=akshay         --data-split=test --store-dir=20201001_akshay_simtest            --save-test-val-results --checkpoint=checkpoints/20201001_akshay/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=1000

# 2020-10-08
# Best test results network was Expt1: No scheduling with 1e-4 learning rate
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C1         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_C1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C2         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_C2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C3         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_C3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C4         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_C4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C5         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_C5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T1         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_T1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T2         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_T2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T3         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_T3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T4         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_T4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T5         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_T5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G1         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_G1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G2         --data-split=test --store-dir=20201008_arun_2_nolrscheduler_G2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2_nolrscheduler/best_model.pt   --architecture=unetsar --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150

# Best test results network  ## 2020-10-10 - Expt1: No scheduling with 1e-4 learning rate with my UNet
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C1_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_C1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C2_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_C2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C3_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_C3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C4_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_C4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C5_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_C5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T1_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_T1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T2_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_T2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T3_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_T3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T4_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_T4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T5_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_T5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G1_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_G1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G2_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_G2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150

# Best test results network  ## 2020-10-10 - Expt2: L1+FFTLoss
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C1_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_C1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C2_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_C2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C3_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_C3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C4_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_C4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C5_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_C5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T1_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_T1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T2_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_T2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T3_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_T3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T4_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_T4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T5_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_T5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G1_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_G1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G2_1024         --data-split=test --store-dir=20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss_G2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_3_realtestdata_onlyfirsttwoseqs_fftloss/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150

## 2020-10-10 - Expt3: 2DCNN FastTimeFirst
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C1_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_C1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C2_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_C2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C3_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_C3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C4_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_C4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C5_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_C5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T1_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_T1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T2_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_T2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T3_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_T3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T4_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_T4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T5_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_T5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G1_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_G1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G2_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst_G2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_fasttimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150

## 2020-10-10 - Expt3: 2DCNN SlowTimeFirst
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C1_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_C1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C2_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_C2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C3_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_C3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C4_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_C4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_C5_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_C5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T1_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_T1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T2_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_T2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T3_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_T3  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T4_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_T4  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_T5_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_T5  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G1_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_G1  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --dataset=real_G2_2D         --data-split=test --store-dir=20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst_G2  --save-test-val-results --checkpoint=checkpoints/20201008_arun_2D_realtestdata_onlyfirsttwoseqs_slowtimefirst/best_model.pt   --architecture=unet2d_fastfirst_3 --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150

# 2020-11-02
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun_interference_testonfirsttwoseqs_-15         --data-split=test --store-dir=20201008_arun_interference_testonfirsttwoseqs_-15  --save-test-val-results --checkpoint=checkpoints/20201102_arun_interference_testonfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun_interference_testonfirsttwoseqs_0           --data-split=test --store-dir=20201008_arun_interference_testonfirsttwoseqs_0    --save-test-val-results --checkpoint=checkpoints/20201102_arun_interference_testonfirsttwoseqs/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_50         --data-split=test --store-dir=20201008_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_50  --save-test-val-results --checkpoint=checkpoints/20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150
# CUDA_VISIBLE_DEVICES=3 python test.py --gpu-ids 0 --dataset=arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_50         --data-split=test --store-dir=20201008_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_50  --save-test-val-results --checkpoint=checkpoints/20201102_arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps/best_model.pt   --architecture=unetsar_arun --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=150

import pdb
import os, sys, glob
import argparse, pathlib
import time
import shutil
import sklearn
# PyTorch DNN imports
import torch
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
# from data import create_dataset_akshay, create_dataset_arun, create_dataset_real, create_dataset_arun_2D, create_dataset_real_2D
from utils.dataset_deets import dataset_deets

# Dataloader imports
from torch.utils.data import DataLoader
from torchvision import transforms
# from utils import MyDataset, MyDataset_Kaz_Training_Raw, ApplySTFT, ApplyPacketLoss, ApplyTelecomDistortions, RandomSubsample, TFGapFillingMaskEstimation, ConcatMaskToInput, ToTensor
# from utils import PoseInterpolatorFCDataset, PoseInterpolatorCNNDataset, ToTensor

# Validation/Testing data related inputs
# from utils import eval_net
from utils import eval_net, write_imgs

# # Import NVIDIA AMP for mixed precision arithmetic
# try:
#     # sys.path.append('/home/t-arnair/Programs/apex')
#     from apex import amp
#     APEX_AVAILABLE = True
#     # OPT_LEVEL = "O2"
#     OPT_LEVEL = "O1"
# except ModuleNotFoundError:
#     APEX_AVAILABLE = False
# # TO OVERRIDE AND JUST IGNORE APEX if necessary
# # APEX_AVAILABLE = False

# NOTE: To set GPU Visibility from inside the code
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# -------------------------------------------------------------------------------------------------
# # Section I - evaluate and visualize functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Testing function from fastMRI code
# def run_unet(args, model, data_loader):
#     model.eval()
#     reconstructions = defaultdict(list)
#     with torch.no_grad():
#         for (input, mean, std, fnames, slices) in data_loader:
#             input = input.unsqueeze(1).to(args.device)
#             recons = model(input).to('cpu').squeeze(1)
#             for i in range(recons.shape[0]):
#                 recons[i] = recons[i] * std[i] + mean[i]
#                 reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

#     reconstructions = {
#         fname: np.stack([pred for _, pred in sorted(slice_preds)])
#         for fname, slice_preds in reconstructions.items()
#     }
#     return reconstructions
#
# After testing using run_unet above, the reconstructions are written to file using save_reconstructions below
# def save_reconstructions(reconstructions, out_dir):
#     """
#     Saves the reconstructions from a model into h5 files that is appropriate for submission
#     to the leaderboard.

#     Args:
#         reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
#             corresponding reconstructions (of shape num_slices x height x width).
#         out_dir (pathlib.Path): Path to the output directory where the reconstructions
#             should be saved.
#     """
#     out_dir.mkdir(exist_ok=True)
#     for fname, recons in reconstructions.items():
#         with h5py.File(out_dir / fname, 'w') as f:
#             f.create_dataset('reconstruction', data=recons)

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
    write_imgs(g_net, params, data_string, data_loader)
    
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
    mean_loss = _loss
    # return np.mean(losses), time.perf_counter() - start
    return mean_points, mean_loss, time.perf_counter() - start

# -------------------------------------------------------------------------------------------------
## Visualization function for val data - outputs it using a tensorboard writer to visually track progress
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

            save_image(target_output, f"Images/Target_{params['data_split']}", nrow=1)
            save_image(preds, f"Images/Reconstruction_{params['data_split']}", nrow=1)
            save_image(torch.abs(target_output - preds), f"Images/Error_{params['data_split']}", nrow=1)
            break

# -------------------------------------------------------------------------------------------------
# # Section II - Dataset definition and dataloader creation
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Create dataset objects to load the data
def create_datasets(params):
    # #--------------------------------------------------------------------------
    # # # COMMENTING THIS SECTION OUT
    # # print('-'*80)
    # # print('Retrieve dataset details...')
    # # print('-'*80)
    # # dataset_deets = retrieve_dataset_filenames(dataset=params['dataset'])
    
    # # print('-'*30)
    # # print('Reading in data filenames...')
    # # print('-'*30)
    # # split_clean_ = []
    # # split_noisy_ = []
    # # for clean_data_path, noisy_data_path in zip(dataset_deets['clean_'+params['data_split']+'_data_path_list'], dataset_deets['noisy_'+params['data_split']+'_data_path_list']):
    # #     curr_data_filenames_clean = sorted(glob.glob(os.path.join(clean_data_path,'*.wav')))
    # #     curr_data_filenames_noisy = glob.glob(os.path.join(noisy_data_path,'*.wav'))
    # #     curr_data_filenames_noisy.sort(key = lambda x: x.split('fileid')[1])
    # #     split_clean_.extend(curr_data_filenames_clean)
    # #     split_noisy_.extend(curr_data_filenames_noisy)
    
    # # # If there is no loaded data to denoise
    # # if not split_noisy_:
    # #     print(f"No data corresponding to chosen data-split of {params['data_split']}_noisy. Exiting.")
    # #     sys.exit(0)

    # # apply_on_clean = True if params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanvoip', 'DNS-challenge-synthetic-test-expanded-cleanteledonline']  else False
    # # center_gap = True if params['architecture'] == 'unetseparable_uros_small_5' else False
    # # _transforms_list =[]
    # # # First, decide if you want to include gap filling
    # # if params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanvoip', 'DNS-challenge-synthetic-test-expanded-noisyvoip']:
    # #     _transforms_list.append(ApplyPacketLoss(apply_on_clean=apply_on_clean, pass_mask=params['pass_mask'], average_init=params['average_init']))
    # # # Second, decide if you want to include telecom distortions generated online
    # # if params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanteledonline', 'DNS-challenge-synthetic-test-expanded-noisyteledonline']:
    # #     _transforms_list.append(ApplyTelecomDistortions(apply_on_clean=apply_on_clean))
    # # if params['architecture'] in params['time_frequency_domain_architectures']:
    # #     _transforms_list.append(ApplySTFT(is_training=False))
    # #     if params['pass_mask']:
    # #         _transforms_list.append(TFGapFillingMaskEstimation())
    # #         _transforms_list.append(ConcatMaskToInput(domain='TF'))
    # # elif params['architecture'] in params['time_domain_architectures']:
    # #     if params['pass_mask']:
    # #         _transforms_list.append(ConcatMaskToInput(domain='T'))
    # # _transforms_list.append(ToTensor())

    # # chosen_dataset   = MyDataset(clean_paths=split_clean_, noisy_paths=split_noisy_, is_training=False,
    # #                         transform=transforms.Compose(_transforms_list))

    # # req_filenames_dict = {f"{params['data_split']}_clean": split_clean_, f"{params['data_split']}_noisy": split_noisy_}
    # # return chosen_dataset, dataset_deets, req_filenames_dict
    # #--------------------------------------------------------------------------
    # if params['dataset']=='akshay':
    #     chosen_dataset = create_dataset_akshay(params, dataset_size=6250, dataset_name = f"{params['data_split']}_set_akshay.pkl")
    # elif params['dataset']=='arun':
    #     chosen_dataset = create_dataset_arun(params, dataset_size=6250, dataset_name = f"{params['data_split']}_set_arun.pkl")
    # elif params['dataset']=='arun_realtestdata':
    #     chosen_dataset = create_dataset_real(params, dataset_name = f"{params['data_split']}_set_real.pkl")
    # elif params['dataset']=='arun_realtestdata_onlyfirsttwoseqs':
    #     chosen_dataset = create_dataset_real(params, dataset_name = f"{params['data_split']}_set_real_onlyfirsttwoseqs.pkl")
    # elif params['dataset'] in ['real_C1', 'real_C2', 'real_C3', 'real_C4', 'real_C5', 'real_T1', 'real_T2', 'real_T3', 'real_T4', 'real_T5', 'real_G1', 'real_G2']:
    #     chosen_dataset = create_dataset_real(params, dataset_name = f"{params['data_split']}_set_{params['dataset']}.pkl")
    # elif params['dataset']=='arun_2D':
    #     chosen_dataset   = create_dataset_arun_2D(params, dataset_size=6250,  dataset_name = f"{params['data_split']}_set_arun_2D.pkl")
    # elif params['dataset']=='arun_2D_realtestdata_onlyfirsttwoseqs':
    #     chosen_dataset   = create_dataset_real_2D(params, dataset_name = f"{params['data_split']}_set_real_onlyfirsttwoseqs_2D.pkl")
    # elif params['dataset']=='arun_2D_realtestdata':
    #     chosen_dataset   = create_dataset_real_2D(params, dataset_name = f"{params['data_split']}_set_real_2D.pkl")
    # elif params['dataset'] in ['real_C1_2D', 'real_C2_2D', 'real_C3_2D', 'real_C4_2D', 'real_C5_2D', 'real_T1_2D', 'real_T2_2D', 'real_T3_2D', 'real_T4_2D', 'real_T5_2D', 'real_G1_2D', 'real_G2_2D']:
    #     chosen_dataset = create_dataset_real_2D(params, dataset_name = f"{params['data_split']}_set_{params['dataset']}.pkl")
    # return chosen_dataset
    _, val_data, test_data = dataset_deets(params)
    if params['data_split']=='val':
        chosen_dataset = val_data
    elif params['data_split']=='test':
        chosen_dataset = test_data
    return chosen_dataset

# -------------------------------------------------------------------------------------------------
## Create dataloaders to load data from my dataset objects
def create_data_loaders(params):
    # mask_func = None
    # if args.mask_kspace:
    #     mask_func = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    # data = SliceData(
    #     root=args.data_path / f'{args.challenge}_{args.data_split}',
    #     transform=DataTransform(args.resolution, args.challenge, mask_func),
    #     sample_rate=1.,
    #     challenge=args.challenge
    # )
    # data_loader = DataLoader(
    #     dataset=data,
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    # return data_loader    
    
    # chosen_dataset, dataset_deets, req_filenames_dict = create_datasets(params)
    chosen_dataset = create_datasets(params)

    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)] 
    if params['no_visualization']:
        display_data_skipfactor = 1
        display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // display_data_skipfactor)]
    else:
        display_data_skipfactor = 20
        # display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // 64)] 
        display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // display_data_skipfactor)] 
    
    # data_loader = torch.utils.data.DataLoader(dataset=chosen_dataset, batch_size=params['test_batch_size'],
    #                             num_workers= 10*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    data_loader = torch.utils.data.DataLoader(chosen_dataset,  batch_size=params['test_batch_size'], shuffle=False)
    display_loader = torch.utils.data.DataLoader(display_data, batch_size=display_data_skipfactor,   shuffle=False)
    
    return data_loader, display_loader

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
    parser.add_argument('--data-split', choices=['val', 'test'], required=True, type=str,
                        help='Which data partition to run on: "val" or "test".')
    # IMPORTANT: usage --store-dir=20190102_20200102_Sim1
    parser.add_argument('--store-dir', required=True, type=pathlib.Path,
                        help='Name of output directory')
    parser.add_argument('--save-test-val-results', action='store_true', default=False,
                        help='Whether to save val and test outputs in mat files')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to an existing checkpoint. Required for testing.')
    parser.add_argument('--architecture', choices=['unetsar', 'unetsar_arun', 'unet2d_fastfirst_3', 'unet2d_slowfirst_3'], default='unetsar_arun', type=str,
                        help='unetsar|unetsar_arun|unet2d_fastfirst_3|unet2d_slowfirst_3|...')
    ## Parameters to update less often
    parser.add_argument('--no-visualization', action='store_true', default=False,
                        help='Disables visualization of the outputs; Also adjusts display_data step size to prevent errors')
    parser.add_argument('--no-neuron-visualization', action='store_true', default=False,
                        help='Disables visualization of the neurons ')
    parser.add_argument('--no-model-copy', action='store_true', default=False,
                        help='Disables copying the model to the results directory')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N_T',
                        help='Input mini-batch size for testing (default: 1000)')
    # parser.add_argument('--test-rows', type=int, default=128, metavar='R',
    #                     help='Number of rows in the test data inputs (default: 128)')
    # parser.add_argument('--test-cols', type=int, default=128, metavar='C',
    #                     help='Number of cols in the test data inputs (default: 128)')
    return parser.parse_args(args)

# -------------------------------------------------------------------------------------------------
## Convert parsed arguments to a dict for more easy manipulation
def create_params_dict(parsed_args, device):
    params = {}
    params['gpu_ids']                   = parsed_args.gpu_ids                     # ids of GPUs being used
    params['dataset']                   = parsed_args.dataset                     # Which dataset to processs
    params['data_split']                = parsed_args.data_split                  # Which data partition to run on: "val" or "test"
    params['results_dir']               = os.path.join("results",parsed_args.store_dir)       # Directory in which to store the output images as mat files
    params['save_test_val_results']     = parsed_args.save_test_val_results       # Whether to save val and test outputs as .mat files
    params['checkpoint']                = parsed_args.checkpoint                  # Directory from which the model is being loaded if its being loaded; false otherwise
    params['architecture']              = parsed_args.architecture                # Set to 'og', 'fair', 'unetseparable_uros', 'unetseparable', 'unet1dk5s2', or 'unet1dk15s4' - determines the neural network architecture implementation
    params['no_visualization']          = parsed_args.no_visualization            # Whether to run visualization of the neurons and outputs; Also adjusts display_data step size to prevent errors
    params['no_neuron_visualization']   = parsed_args.no_neuron_visualization     # Whether to run visualization of the neurons and outputs; Also adjusts display_data step size to prevent errors
    params['no_model_copy']             = parsed_args.no_model_copy               # Whether to disable copying the model to the results directory    
    params['test_batch_size']           = parsed_args.test_batch_size             # Number of testing files in one mini-batch - can be much larger since gradient information isn't required
    # params['test_rows']                 = parsed_args.test_rows                   # Number of rows in the test images
    # params['test_cols']                 = parsed_args.test_cols                   # Number of columns in the test images
    params['device']                    = device                                  # Device to run the code on

    return params

# -------------------------------------------------------------------------------------------------
# # Section IV - Smaller Utility Functions
# -------------------------------------------------------------------------------------------------

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
    
    if not parsed_args.no_parallel:
        # Set it to parallelize across visible GPUs
        g_net = torch.nn.DataParallel(g_net)
    
    # Move the model to the GPUs
    g_net.to(device)
    
    return g_net

# -------------------------------------------------------------------------------------------------
## Load the DNN model from disk
def load_model(parsed_args, device):
    checkpoint = torch.load(parsed_args.checkpoint)
    print(f"Generator Model loaded from {parsed_args.checkpoint}")
    
    parsed_args_model = checkpoint['parsed_args']
    g_net = build_model(parsed_args_model, device)
    g_net.load_state_dict(checkpoint['g_net'])

    return checkpoint, g_net, parsed_args_model.criterion_g, parsed_args_model.no_parallel

# -------------------------------------------------------------------------------------------------
## Initialize the loss criteria
def initialize_loss_criterion(params):
    # Options for possible loss functions
    # loss_dict = {'dscloss':DiceCoeffLoss(), 'maeloss': torch.nn.L1Loss(), 'mseloss': torch.nn.MSELoss(), 'rmseloss': RMSELoss(), 'lsdloss': LSDLoss(), 'bceloss': torch.nn.BCELoss()}
    loss_dict = {'l1loss': torch.nn.L1Loss(), 'smoothl1loss': torch.nn.SmoothL1Loss(), 'mseloss': torch.nn.MSELoss(), 'l1andfftloss': L1andFFTLoss}
    # NOTE: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    # So use it in general when possible...
    
    # Set loss functionsn to use for training DNN
    criterion_g = loss_dict[params['criterion_g']]
    # if params['criterion_g']=='l1loss':
    #     criterion_g = loss_dict['maeloss']
    # else:    
    #     criterion_g = loss_dict[params['criterion_g']]

    # Moving it to the GPU - not really required except for stateful losses as stated in https://discuss.pytorch.org/t/move-the-loss-function-to-gpu/20060
    # criterion_g.to(params['device'])

    return criterion_g

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

    # Create a dict out of the parsed_args
    params = create_params_dict(parsed_args, device)

    print('-'*80)
    print('Create the data loaders...')
    print('-'*80)
    data_loader, display_loader = create_data_loaders(params)

    print('-'*80)
    print('Load the model from disk...')
    print('-'*80)
    checkpoint, g_net, params['criterion_g'], params['no_parallel'] = load_model(parsed_args, device)

    model_epoch = checkpoint['epoch']
    del checkpoint

    criterion_g = initialize_loss_criterion(params)
    writer = SummaryWriter(log_dir = os.path.join(params['results_dir'], 'summary'))

    # Do the validation/test step
    _points, _loss, _time = evaluate(params, model_epoch, g_net, criterion_g, data_loader, writer, params['data_split'])
    
    if not params['no_neuron_visualization']:
        visualize_neurons(params, model_epoch, g_net, display_loader, writer)
    if not params['no_visualization']:
        visualize(params, model_epoch, g_net, display_loader, writer)

    os.makedirs(os.path.join(params['results_dir'],'model'), exist_ok=True)
    if not params['no_model_copy']:
        # Copy the run model file into the results directory        
        shutil.copyfile(parsed_args.checkpoint, os.path.join(os.path.join(params['results_dir'],'model','copied_model.pt')))    

    # Save the test argparse into the results directory
    save_file_string  = os.path.join(params['results_dir'], 'testing_deets.txt')
    with open(save_file_string, 'w+') as f: # we don't have to write "file.close()". That will automatically be called.
        for key, value in params.items():
            f.write(f"{key}:{value}\n")

    # Write network graph to file
    # NOTE: 1) Doesn't work with dataparallel 2) Need to use the FAIR UNet code
    # sample = next(iter(display_loader))
    # writer.add_graph(g_net, sample['input_data'].to(device))

    writer.close()
    del g_net
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main(sys.argv[1:])
