class CFG:
    release_mode = 'inference' # None, 'inference', 'retrain'
    DB_PATH = '/home/doinglab-hs/ak/db/eng_df.csv'

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
    MAX_EPOCH = 1
    BATCH_SIZE = 8
    NUM_WORKS = 8
    
    # Re Training
    CON_TRAINING_CHEKPOINT = './ckpt/con_epoch_0.pth'
    CE_TRAINING_CHEKPOINT = './ckpt/ce_epoch_0.pth'
    
    ### INFERNCE_PARAM ###
    BEST_CON_PATH = './ckpt/best_con.pth'
    BEST_CE_PATH = './ckpt/best_ce.pth'