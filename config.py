mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "How2Sign": "./data/How2Sign/labels.train",
                    "OpenASL": "./data/OpenASL/labels.train",
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "How2Sign": "",
                    "OpenASL": "./data/OpenASL/labels.dev",
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "How2Sign": "./data/How2Sign/labels.test",
                    "OpenASL": "./data/OpenASL/labels.test",
}


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "How2Sign": "./dataset/How2Sign/rgb_format",
            "OpenASL": "./dataset/OpenASL/rgb_format",
            }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "How2Sign": "./dataset/WLASL/pose_format",
            "OpenASL": "./dataset/WLASL/pose_format",
}

# Pose Tokenizer Configuration
POSE_TOKENIZER_CONFIG = {
    # Whether to use pose tokenizer
    'use_pose_tokenizer': False,

    # Tokenizer architecture parameters
    'tokenizer_hidden_dim': 256,
    'num_quantizers': 4,
    'codebook_size': 1024,
    'commitment_cost': 0.25,

    # Training parameters
    'vq_loss_weight': 1.0,
    'tokenizer_lr': 1e-4,
    'tokenizer_warmup_epochs': 5,

    # Evaluation parameters
    'log_codebook_usage': True,
    'save_quantized_features': False,
}