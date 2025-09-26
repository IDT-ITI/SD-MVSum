from model.layers.summarizer import SD_VSum
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import os


class BaseSolver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None):
        self.model, self.optimizer = None, None
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.BCELoss()

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        self.model = SD_VSum(
            input_size=self.config.input_size,
            text_size=self.config.text_size,
            output_size=self.config.input_size,
            heads=self.config.heads,
            pos_enc=self.config.pos_enc,
            visual_weights=self.config.visual_weights,
            transcript_weights=self.config.transcript_weights
        ).to(self.config.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.l2_reg
        )

        if os.path.exists(self.config.ckpt_path):
            print(f"Loading pretrained model from {self.config.ckpt_path}")
            self.model.load_state_dict(torch.load(self.config.ckpt_path))
        else:
            print("Training from scratch")
            if self.config.init_type is not None:
                self.init_weights(self.model, self.config.init_type, self.config.init_gain)

    def test(self, ckpt_path):
        """Generic test method: loads model checkpoint and evaluates."""
        print(f"Loading checkpoint from {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.config.device))
        self.model.to(self.config.device)

        print("Running evaluation on test set...")
        results = self.evaluate(dataloader=self.test_loader)
        print("Test results:", results)

        # Optionally log results
        with open(os.path.join(self.config.save_dir_root, 'test_results.txt'), 'w') as f:
            f.write(str(results))

        return results

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                else:
                    raise NotImplementedError(f"Init {init_type} not implemented")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)