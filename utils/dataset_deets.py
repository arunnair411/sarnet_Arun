import pdb
from data import create_dataset_akshay, create_dataset_arun, create_dataset_real, create_dataset_arun_2D, create_dataset_real_2D, create_dataset_arun_testdistributed, create_dataset_arun_testdistributedandregular, create_arun_testdistributed_CTsplittrain_CTsplittest, create_arun_testdistributed_CTsplittrain_CTsplittest_interference, create_arun_interference, create_dataset_arun_fwd, create_dataset_arun_multiplemissingrates, generic_dataset_loader, create_dataset_arun_generative

def dataset_deets(params):
    if params['dataset']=='akshay':
        # 1 - Akshay's original training data - generated using a dictionary and sparse code
        train_data = create_dataset_akshay(params, dataset_size=50000, dataset_name = 'train_set_akshay.pkl')
        val_data   = create_dataset_akshay(params, dataset_size=6250,  dataset_name = 'val_set_akshay.pkl')
        test_data  = create_dataset_akshay(params, dataset_size=6250,  dataset_name = 'test_set_akshay.pkl')
    elif params['dataset']=='arun':
        # 2 - My original attempt at creating new data - generated using a radar simulator - train and test on that dataset
        train_data = create_dataset_arun(params, dataset_size=50000, dataset_name = 'train_set_arun.pkl')
        val_data   = create_dataset_arun(params, dataset_size=6250,  dataset_name = 'val_set_arun.pkl')
        test_data  = create_dataset_arun(params, dataset_size=6250,  dataset_name = 'test_set_arun.pkl')
    elif params['dataset']=='arun_realtestdata_onlyfirsttwoseqs':
        # 3 - 2, except I switch the test set to be the first 2 sequences (C1.mat and C2.mat) of the real test data
        # Have alternate code put in here to generate 1000x1 data for akshay's neural network instead of mine
        train_data = create_dataset_arun(params, dataset_size=50000, dataset_name = 'train_set_arun.pkl')
        # train_data = create_dataset_arun(params, dataset_size=50000, dataset_name = 'akshay_train_set_arun.pkl', line_length=1000)
        val_data   = create_dataset_arun(params, dataset_size=6250,  dataset_name = 'val_set_arun.pkl')
        # val_data   = create_dataset_arun(params, dataset_size=6250,  dataset_name = 'akshay_val_set_arun.pkl', line_length=1000)
        test_data  = create_dataset_real(params, dataset_name = 'test_set_real_onlyfirsttwoseqs.pkl')
        # test_data  = create_dataset_real(params, dataset_name = 'akshay_test_set_real_onlyfirsttwoseqs.pkl', line_length=1000)
    elif params['dataset']=='arun_realtestdata':
        # 4 - 3, except I switch the test set to be all sequences instead of just the first 2
        train_data = create_dataset_arun(params, dataset_size=50000, dataset_name = 'train_set_arun.pkl')
        val_data   = create_dataset_arun(params, dataset_size=6250,  dataset_name = 'val_set_arun.pkl')
        test_data  = create_dataset_real(params, dataset_name = 'test_set_real.pkl')
    elif params['dataset']=='arun_testdistributed':
        # 5 - Using sparse code information from the test set to train, and testing on the test set itself - cheating one
        train_data = create_dataset_arun_testdistributed(params, dataset_name = 'train_set_arun_testdistributed.pkl')
        val_data   = create_dataset_arun_testdistributed(params, dataset_name = 'val_set_arun_testdistributed.pkl')
        test_data  = create_dataset_real(params, dataset_name = 'test_set_real.pkl')    
    elif params['dataset']=='arun_testdistributed_CTsplittrain_CTsplittest':
        # 6 - Using sparse code information from a split of the test set to train, and testing on the other split
        train_data = create_arun_testdistributed_CTsplittrain_CTsplittest(params, with_regular = False, dataset_name = 'train_CTsplit_set_arun_testdistributed.pkl')
        val_data   = create_arun_testdistributed_CTsplittrain_CTsplittest(params, with_regular = False, dataset_name = 'val_CTsplit_set_arun_testdistributed.pkl')
        test_data  = create_arun_testdistributed_CTsplittrain_CTsplittest(params, with_regular = False, dataset_name = 'test_CTsplit_set_arun_testdistributed.pkl')
    elif params['dataset']=='arun_testdistributed_CTsplittrain_CTsplittestandregular':
        # 7 - Using sparse code information from a split of the test set + simulated data to train, and testing on the other split
        train_data = create_arun_testdistributed_CTsplittrain_CTsplittest(params, with_regular = True,  dataset_name = 'train_CTsplit_set_arun_testdistributed.pkl')
        val_data   = create_arun_testdistributed_CTsplittrain_CTsplittest(params, with_regular = False, dataset_name = 'val_CTsplit_set_arun_testdistributed.pkl')
        test_data  = create_arun_testdistributed_CTsplittrain_CTsplittest(params, with_regular = False, dataset_name = 'test_CTsplit_set_arun_testdistributed.pkl')
    # Interference ##################################################################################################################
    elif params['dataset']=='arun_interference':
        # 8 - Using sparse code information from a split of the test set to train, and testing on the other split - with interference
        train_data = create_arun_interference(params, dataset_name = 'train_interference_set_arun.pkl')
        val_data   = create_arun_interference(params, dataset_name = 'val_interference_set_arun.pkl')
        test_data  = create_arun_interference(params, dataset_name = 'test_interference_set_arun.pkl')
    elif params['dataset']=='arun_interference_testonfirsttwoseqs':
        # 8 - Using sparse code information from a split of the test set to train, and testing on the other split - with interference
        train_data = create_arun_interference(params, dataset_name = 'train_interference_set_arun.pkl')
        val_data   = create_arun_interference(params, dataset_name = 'val_interference_set_arun.pkl')
        test_data  = create_arun_interference(params, dataset_name = 'test_interference_set_real_onlyfirsttwoseqs.pkl')
    elif params['dataset'] in ['arun_interference_testonfirsttwoseqs_-15', 'arun_interference_testonfirsttwoseqs_-10', 
    'arun_interference_testonfirsttwoseqs_-5', 'arun_interference_testonfirsttwoseqs_0', 'arun_interference_testonfirsttwoseqs_5', 'arun_interference_testonfirsttwoseqs_10']:
        # 8 - Using sparse code information from a split of the test set to train, and testing on the other split - with interference
        train_data = create_arun_interference(params, dataset_name = 'train_interference_set_arun.pkl')
        snr_string = params['dataset'].split('_')[-1]
        val_data   = create_arun_interference(params, dataset_name = 'val_interference_set_arun_'+ snr_string +'.pkl')
        test_data  = create_arun_interference(params, dataset_name = 'test_interference_set_real_onlyfirsttwoseqs_' + snr_string + '.pkl')
    elif params['dataset']=='arun_testdistributed_CTsplittrain_CTsplittest_interference':
        # 8 - Using sparse code information from a split of the test set to train, and testing on the other split - with interference
        train_data = create_arun_testdistributed_CTsplittrain_CTsplittest_interference(params, with_regular = False, dataset_name = 'train_interference_CTsplit_set_arun_testdistributed.pkl')
        val_data   = create_arun_testdistributed_CTsplittrain_CTsplittest_interference(params, with_regular = False, dataset_name = 'val_interference_CTsplit_set_arun_testdistributed.pkl')
        test_data  = create_arun_testdistributed_CTsplittrain_CTsplittest_interference(params, with_regular = False, dataset_name = 'test_interference_CTsplit_set_arun_testdistributed.pkl')
    elif params['dataset']=='arun_testdistributed_CTsplittrain_CTsplittestandregular_interference':
        # 9 - Using sparse code information from a split of the test set + simulated data to train, and testing on the other split - with interference
        train_data = create_arun_testdistributed_CTsplittrain_CTsplittest_interference(params, with_regular = True,  dataset_name = 'train_interference_CTsplit_set_arun_testdistributed.pkl')
        val_data   = create_arun_testdistributed_CTsplittrain_CTsplittest_interference(params, with_regular = False, dataset_name = 'val_interference_CTsplit_set_arun_testdistributed.pkl')
        test_data  = create_arun_testdistributed_CTsplittrain_CTsplittest_interference(params, with_regular = False, dataset_name = 'test_interference_CTsplit_set_arun_testdistributed.pkl')        
    elif params['dataset']=='arun_testdistributedandregular':
        train_data = create_dataset_arun_testdistributedandregular(params, dataset_name = 'train_set_arun.pkl')
        val_data   = create_dataset_arun_testdistributed(params, dataset_name = 'val_set_arun.pkl')
        test_data  = create_dataset_real(params, dataset_name = 'test_set_real.pkl')
    # elif params['dataset']=='arun_testdistributed_CTsplittrain_CTsplittest_andregular':
    #     train_data = create_dataset_arun_CTtestdistributedandregular(params, dataset_name = 'train_set_arun.pkl')
    #     val_data   = create_dataset_arun_CTtestdistributedandregular(params, dataset_name = 'val_set_arun.pkl')
    #     test_data  = create_dataset_real(params, dataset_name = 'test_set_real_CTsplit.pkl')        
    # elif params['dataset']=='arun_interference_testdistributed_CTsplittrain_CTsplittest_andregular':
    #     train_data = create_dataset_arun_CTtestdistributedandregular(params, dataset_name = 'train_interference_set_arun.pkl')
    #     val_data   = create_dataset_arun_CTtestdistributedandregular(params, dataset_name = 'val_interference_set_arun.pkl')
    #     test_data  = create_dataset_real(params, dataset_name = 'test_set_real_CTsplit.pkl') # TODO: Update this
####################################################
    elif params['dataset']=='arun_2D':
        train_data = create_dataset_arun_2D(params, dataset_size=50000, dataset_name = 'train_set_arun_2D.pkl')
        val_data   = create_dataset_arun_2D(params, dataset_size=6250,  dataset_name = 'val_set_arun_2D.pkl')
        test_data  = create_dataset_arun_2D(params, dataset_size=6250,  dataset_name = 'test_set_arun_2D.pkl')
    elif params['dataset']=='arun_2D_realtestdata_onlyfirsttwoseqs':
        train_data = create_dataset_arun_2D(params, dataset_size=50000, dataset_name = 'train_set_arun_2D.pkl')
        val_data   = create_dataset_arun_2D(params, dataset_size=6250,  dataset_name = 'val_set_arun_2D.pkl')
        test_data  = create_dataset_real_2D(params, dataset_name = 'test_set_real_onlyfirsttwoseqs_2D.pkl')
    elif params['dataset']=='arun_2D_realtestdata':
        train_data = create_dataset_arun_2D(params, dataset_size=50000, dataset_name = 'train_set_arun_2D.pkl')
        val_data   = create_dataset_arun_2D(params, dataset_size=6250,  dataset_name = 'val_set_arun_2D.pkl')
        test_data  = create_dataset_real_2D(params, dataset_name = 'test_set_real_2D.pkl')        
####################################################
    elif params['dataset']=='arun_fwd_exact':
        train_data = create_dataset_arun_fwd(params, dataset_name = 'train_fwd_exact_set_arun.pkl')
        val_data   = create_dataset_arun_fwd(params, dataset_name = 'val_fwd_set_arun.pkl')
        test_data  = create_dataset_arun_fwd(params, dataset_name = 'test_fwd_set_arun.pkl')        
    elif params['dataset']=='arun_fwd_generated':
        train_data = create_dataset_arun_fwd(params, dataset_name = 'train_fwd_generated_set_arun.pkl')
        val_data   = create_dataset_arun_fwd(params, dataset_name = 'val_fwd_set_arun.pkl')
        test_data  = create_dataset_arun_fwd(params, dataset_name = 'test_fwd_set_arun.pkl')                
    elif params['dataset']=='arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates':
        train_data = create_dataset_arun_multiplemissingrates(params, dataset_name = 'train_set_arun_multiplemissingrates.pkl')
        val_data   = create_dataset_arun_multiplemissingrates(params, dataset_name = 'val_set_arun_multiplemissingrates.pkl')
        test_data  = create_dataset_arun_multiplemissingrates(params, dataset_name = 'test_real_onlyfirsttwoseqs_set_arun_multiplemissingrates.pkl')
    elif params['dataset'] in ['arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_50', 'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_60',
    'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_70', 'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_80', 'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_90']:
        train_data = create_dataset_arun_multiplemissingrates(params, dataset_name = 'train_set_arun_multiplemissingrates.pkl')
        missing_percentage = params['dataset'].split('_')[-1]
        val_data   = create_dataset_arun_multiplemissingrates(params, dataset_name = 'val_set_arun_multiplemissingrates_' + missing_percentage + '.pkl')
        test_data  = create_dataset_arun_multiplemissingrates(params, dataset_name = 'test_real_onlyfirsttwoseqs_set_arun_multiplemissingrates_' + missing_percentage + '.pkl')
    elif params['dataset']=='arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps':
        train_data = create_dataset_arun_multiplemissingrates(params, dataset_name = 'train_set_arun_multiplemissingrates_randomgaps.pkl')
        val_data   = create_dataset_arun_multiplemissingrates(params, dataset_name = 'val_set_arun_multiplemissingrates_randomgaps.pkl')        
        test_data  = create_dataset_arun_multiplemissingrates(params, dataset_name = 'test_real_onlyfirsttwoseqs_set_arun_multiplemissingrates_randomgaps.pkl')        
    elif params['dataset'] in ['arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_50', 'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_60',
    'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_70', 'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_80', 'arun_realtestdata_onlyfirsttwoseqs_multiplemissingrates_randomgamps_90']:
        train_data = create_dataset_arun_multiplemissingrates(params, dataset_name = 'train_set_arun_multiplemissingrates_randomgaps.pkl')
        missing_percentage = params['dataset'].split('_')[-1]
        val_data   = create_dataset_arun_multiplemissingrates(params, dataset_name = 'val_set_arun_multiplemissingrates_randomgaps_' + missing_percentage + '.pkl')
        test_data  = create_dataset_arun_multiplemissingrates(params, dataset_name = 'test_real_onlyfirsttwoseqs_set_arun_multiplemissingrates_randomgaps_' + missing_percentage + '.pkl')        
    elif params['dataset']=='arun_generative_modeled_realtestdata_onlyfirsttwoseqs':
        train_data = create_dataset_arun_generative(params, dataset_name = 'train_set_arun_generative_modeled.pkl')
        val_data   = create_dataset_arun_generative(params, dataset_name = 'val_set_arun_generative_modeled.pkl')        
        test_data  = create_dataset_real(params, dataset_name = 'test_set_real_onlyfirsttwoseqs.pkl')
    elif params['dataset']=='arun_extended_and_generative_testononlyfirsttwoseqs':
        train_data = generic_dataset_loader(params, dataset_names = ['train_set_arun_extended.pkl', 'train_set_arun_generative_modeled_extended.pkl'])
        val_data   = generic_dataset_loader(params, dataset_names = ['val_set_arun.pkl', 'val_set_arun_generative_modeled.pkl'])        
        test_data  = create_dataset_real(params, dataset_name = 'test_set_real_onlyfirsttwoseqs.pkl')
    elif params['dataset']=='arun_extended_and_generative_testononlyfirsttwoseqs_interference':
        train_data = generic_dataset_loader(params, dataset_names = ['train_interference_set_arun_extended.pkl', 'train_interference_set_arun_generative_modeled_extended.pkl'])
        val_data   = generic_dataset_loader(params, dataset_names = ['val_interference_set_arun.pkl', 'val_interference_set_arun_generative_modeled.pkl'])
        test_data  = generic_dataset_loader(params, dataset_name = 'test_interference_set_real_onlyfirsttwoseqs.pkl')        
    elif params['dataset']=='arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps':
        train_data = generic_dataset_loader(params, dataset_names = ['train_set_arun_extended_randomgaps_50.pkl', 'train_set_arun_generative_modeled_extended_randomgaps_50.pkl'])
        val_data   = generic_dataset_loader(params, dataset_names = ['val_set_arun_randomgaps_50.pkl', 'val_set_arun_generative_modeled_randomgaps_50.pkl'])
        test_data  = generic_dataset_loader(params, dataset_name = 'test_set_real_onlyfirsttwoseqs_randomgaps_50.pkl')                
    elif params['dataset']=='arun_extended_and_generative_testononlyfirsttwoseqs_randomgaps_allrates':
        train_data = generic_dataset_loader(params, dataset_names = ['train_set_arun_extended_randomgaps_50.pkl', 'train_set_arun_generative_modeled_extended_randomgaps_50.pkl',
         'train_set_arun_extended_randomgaps_60.pkl', 'train_set_arun_generative_modeled_extended_randomgaps_60.pkl',
         'train_set_arun_extended_randomgaps_70.pkl', 'train_set_arun_generative_modeled_extended_randomgaps_70.pkl',
         'train_set_arun_extended_randomgaps_80.pkl', 'train_set_arun_generative_modeled_extended_randomgaps_80.pkl',
         'train_set_arun_extended_randomgaps_90.pkl', 'train_set_arun_generative_modeled_extended_randomgaps_90.pkl'])
        val_data   = generic_dataset_loader(params, dataset_names = ['val_set_arun_randomgaps_50.pkl', 'val_set_arun_generative_modeled_randomgaps_50.pkl',
        'val_set_arun_randomgaps_60.pkl', 'val_set_arun_generative_modeled_randomgaps_60.pkl',
        'val_set_arun_randomgaps_70.pkl', 'val_set_arun_generative_modeled_randomgaps_70.pkl',
        'val_set_arun_randomgaps_80.pkl', 'val_set_arun_generative_modeled_randomgaps_80.pkl',
        'val_set_arun_randomgaps_90.pkl', 'val_set_arun_generative_modeled_randomgaps_90.pkl'])
        test_data  = generic_dataset_loader(params, dataset_names = ['test_set_real_onlyfirsttwoseqs_randomgaps_50.pkl',
        'test_set_real_onlyfirsttwoseqs_randomgaps_60.pkl', 'test_set_real_onlyfirsttwoseqs_randomgaps_70.pkl', 
        'test_set_real_onlyfirsttwoseqs_randomgaps_80.pkl', 'test_set_real_onlyfirsttwoseqs_randomgaps_90.pkl'])
    elif params['dataset']=='arun_extended_and_generative_testononlyfirsttwoseqs_blockgaps_allrates':
        train_data = generic_dataset_loader(params, dataset_names = ['train_set_arun_extended_blockgaps_50.pkl', 'train_set_arun_generative_modeled_extended_blockgaps_50.pkl',
         'train_set_arun_extended_blockgaps_60.pkl', 'train_set_arun_generative_modeled_extended_blockgaps_60.pkl',
         'train_set_arun_extended_blockgaps_70.pkl', 'train_set_arun_generative_modeled_extended_blockgaps_70.pkl',
         'train_set_arun_extended_blockgaps_80.pkl', 'train_set_arun_generative_modeled_extended_blockgaps_80.pkl',
         'train_set_arun_extended_blockgaps_90.pkl', 'train_set_arun_generative_modeled_extended_blockgaps_90.pkl'])
        val_data   = generic_dataset_loader(params, dataset_names = ['val_set_arun_blockgaps_50.pkl', 'val_set_arun_generative_modeled_blockgaps_50.pkl',
        'val_set_arun_blockgaps_60.pkl', 'val_set_arun_generative_modeled_blockgaps_60.pkl',
        'val_set_arun_blockgaps_70.pkl', 'val_set_arun_generative_modeled_blockgaps_70.pkl',
        'val_set_arun_blockgaps_80.pkl', 'val_set_arun_generative_modeled_blockgaps_80.pkl',
        'val_set_arun_blockgaps_90.pkl', 'val_set_arun_generative_modeled_blockgaps_90.pkl'])
        test_data  = generic_dataset_loader(params, dataset_names = ['test_set_real_onlyfirsttwoseqs_blockgaps_50.pkl',
        'test_set_real_onlyfirsttwoseqs_blockgaps_60.pkl', 'test_set_real_onlyfirsttwoseqs_blockgaps_70.pkl', 
        'test_set_real_onlyfirsttwoseqs_blockgaps_80.pkl', 'test_set_real_onlyfirsttwoseqs_blockgaps_90.pkl'])        
    # pdb.set_trace()
    return train_data, val_data, test_data
