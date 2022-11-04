class CFG:
    release_mode = None # None, 'inference', 'retrain_con'

    ### TRAIN_PARAM ###
    # Image Encoder
    RESIZE = 224
    MODEL_NAME = "tf_efficientnetv2_s_in21ft1k" 
    image_embedding = 1280
    
    pretrained = True
    
    # ProjectionHead
    num_projection_layers = 1
    projection_dim = 512
    dropout = 0.1
    
    # SupConLoss
    temperature = 0.07
    
    # Main
    VISIBLE_GPU = "0"
    MAX_EPOCH = 20
    BATCH_SIZE = 8
    NUM_WORKS = 8
    
    # Re Training
    CON_TRAINING_CHEKPOINT = 'D:/my_git/doing/SupCon/ckpt/con_epoch_0.pth'
    CE_TRAINING_CHEKPOINT = 'D:/my_git/doing/SupCon/ckpt/ce_epoch_0.pth'
    
    ### INFERNCE_PARAM ###
    BEST_CON_PATH = 'D:/my_git/doing/SupCon/ckpt/best_con.pth'
    BEST_CE_PATH = 'D:/my_git/doing/SupCon/ckpt/best_ce.pth'