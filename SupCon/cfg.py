class CFG:
    RELEASE_MODE = 'retrain' # None, 'inference', 'retrain'
    LEARNING_STEP = ['second'] # ['first', 'second']

    ### TRAIN_PARAM ###
    # Image Encoder
    RESIZE = 224
    MODEL_NAME = "tf_efficientnetv2_s_in21ft1k" 
    image_embedding = 1280
    
    pretrained = True
    
    # ProjectionHead
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
    ROOT_CHEKPOINT = 'D:/my_git/doing/SupCon/ckpt/'
    CON_TRAINING_CHEKPOINT = ROOT_CHEKPOINT + 'con_epoch_2.pth'
    CE_TRAINING_CHEKPOINT = ROOT_CHEKPOINT + 'ce_epoch_1.pth'
    
    ### INFERNCE_PARAM ###
    BEST_CON_PATH = ROOT_CHEKPOINT + 'best_con.pth'
    BEST_CE_PATH = ROOT_CHEKPOINT + 'best_ce.pth'