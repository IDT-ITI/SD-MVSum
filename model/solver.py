from model.SVideoXum_utils.evaluation_metrics import evaluate_summary as evaluate_svideoxum
from model.SMrhiSum_utils.evaluation_metrics import evaluate_summary as evaluate_smrhisum
from model.SMrhiSum_utils.generate_summary import generate_summary
from model.base_solver import BaseSolver
from tqdm import trange
import numpy as np
import torch
import os


class MrHiSumSolver(BaseSolver):
    def train(self):
        best_f1score = -1.0
        best_f1score_epoch = 0

        loss_total = []
        f1score_total = []

        val_f1score, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)
        f1score_total.append(val_f1score)

        f_score_path = os.path.join(self.config.save_dir_root, "val_f1score.txt")
        with open(f_score_path, "w") as file:
            file.write(f"{val_f1score}\n")

        for epoch_i in range(self.config.epochs):
            print(f"[Epoch: {epoch_i}/{self.config.epochs}]")
            self.model.train()
            loss_history = []

            for batch in trange(len(self.train_loader), desc='Batch', ncols=80, leave=False):
                self.optimizer.zero_grad()
                data = self.train_loader.dataset[batch]

                # unpack batch tuple: MrHiSum always returns 6 elements
                video_features, text_features, transcripts, gtscore, cps, gt_summary = data

                video_features = video_features.to(self.config.device).squeeze()
                text_features = text_features.to(self.config.device).squeeze(0)
                transcripts = transcripts.to(self.config.device).squeeze()
                gtscore = gtscore.to(self.config.device).squeeze(0)


                score = self.model(video_features, text_features, transcripts)
                loss = self.criterion(score.squeeze(0), gtscore)
                loss.backward()
                loss_history.append(loss.detach())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            epoch_loss = torch.stack(loss_history).mean()
            val_f1score, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)

            loss_total.append(epoch_loss)
            f1score_total.append(val_f1score)

            with open(f_score_path, "a") as file:
                file.write(f"{val_f1score}\n")

            if val_f1score >= best_f1score:
                best_f1score = val_f1score
                best_kTau = val_kTau
                best_sRho = val_sRho
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, 'best_f1.pkl')
                torch.save(self.model.state_dict(), f1_save_ckpt_path)

            print(f"   [Epoch {epoch_i}] Train loss: {epoch_loss:.5f}")
            print(f"    VAL  F-score {val_f1score:.5f}")
            print(f"    VAL  kTau {val_kTau:.5f}")
            print(f"    VAL  sRho {val_sRho:.5f}")


        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch))
        f.write('   Best Val kTau {0:0.5} @ epoch{1}\n'.format(best_kTau, best_f1score_epoch))
        f.write('   Best Val kTau {0:0.5} @ epoch{1}\n'.format(best_sRho, best_f1score_epoch))
        f.flush()
        f.close()

        return f1_save_ckpt_path

    def evaluate(self, dataloader=None):
        self.model.eval()
        fscore_history, kTau_history, sRho_history = [], [], []

        for data in dataloader:
            video_features, text_features, transcripts, gtscore, cps, gt_summary = data
            video_features = video_features.to(self.config.device).squeeze()
            text_features = text_features.to(self.config.device).squeeze(0)
            transcripts = transcripts.to(self.config.device).squeeze()
            gtscore = gtscore.squeeze()
            cps = cps.squeeze(0)
            gt_summary = gt_summary.squeeze(0)
            nfps = [(int((cp[1] - cp[0]).item())) for cp in cps]

            with torch.no_grad():
                score = self.model(video_features, text_features, transcripts)
            score = score.squeeze().cpu()

            n_frames = video_features.shape[0]
            picks = np.arange(n_frames)
            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            f_score, kTau, sRho = evaluate_smrhisum(machine_summary, gt_summary, score, gtscore)
            fscore_history.append(f_score)
            kTau_history.append(kTau)
            sRho_history.append(sRho)

        return np.mean(fscore_history), np.mean(kTau_history), np.mean(sRho_history)


class VideoXumSolver(BaseSolver):
    def train(self):
        best_f1score = -1.0
        best_f1score_epoch = 0

        loss_total = []
        f1score_total = []

        val_f1score = self.evaluate(dataloader=self.val_loader)
        f1score_total.append(val_f1score)

        f_score_path = os.path.join(self.config.save_dir_root, "val_f1score.txt")
        with open(f_score_path, "w") as file:
            file.write(f"{val_f1score}\n")

        for epoch_i in range(self.config.epochs):
            print(f"[Epoch: {epoch_i}/{self.config.epochs}]")
            self.model.train()
            loss_history = []

            for batch in trange(len(self.train_loader), desc='Batch', ncols=80, leave=False):
                self.optimizer.zero_grad()
                data = self.train_loader.dataset[batch]

                # unpack batch tuple: VideoXum returns 4 elements
                video_features, text_features, transcripts, gtscore = data
                video_features = video_features.to(self.config.device).squeeze()
                text_features = text_features.to(self.config.device).squeeze(0)
                transcripts = transcripts.to(self.config.device).squeeze()
                gtscores = gtscore.to(self.config.device).squeeze(0)

                for i in range(self.config.annotations):
                    current_gtscore = gtscores[i]
                    current_text_features = text_features[i, :, :]
                    mask = (current_text_features.abs().sum(dim=1) != 0)
                    current_text_features = current_text_features[mask]
                    if current_text_features.shape[0] == 0:
                        continue
                    score = self.model(video_features, current_text_features, transcripts)
                    loss = self.criterion(score.squeeze(0), current_gtscore)
                    loss.backward()
                    loss_history.append(loss.detach())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            epoch_loss = torch.stack(loss_history).mean()
            val_f1score = self.evaluate(dataloader=self.val_loader)

            loss_total.append(epoch_loss)
            f1score_total.append(val_f1score)

            with open(f_score_path, "a") as file:
                file.write(f"{val_f1score}\n")

            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                torch.save(self.model.state_dict(), f1_save_ckpt_path)
            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('    VAL  F-score {0:0.5} '.format(val_f1score))

        return f1_save_ckpt_path

    def evaluate(self, dataloader=None):
        self.model.eval()
        fscore_history = []

        for data in dataloader:
            video_features, text_features, transcripts, gtscore = data
            video_features = video_features.to(self.config.device).squeeze()
            text_features = text_features.to(self.config.device).squeeze(0)
            transcripts = transcripts.to(self.config.device).squeeze()
            gtscores = gtscore.to(self.config.device).squeeze(0)

            fscore_video = []
            for i in range(self.config.annotations):
                current_gtscore = gtscores[i]
                if text_features.ndim == 2:
                    text_features = text_features.unsqueeze(0)
                current_text_features = text_features[i, :, :]
                mask = (current_text_features.abs().sum(dim=1) != 0)
                current_text_features = current_text_features[mask]
                if current_text_features.shape[0] == 0:
                    continue
                with torch.no_grad():
                    score = self.model(video_features, current_text_features, transcripts)
                f_score = evaluate_svideoxum(score.squeeze().cpu(), current_gtscore.cpu())
                fscore_video.append(f_score)

            if fscore_video:
                fscore_history.append(np.mean(fscore_video))

        return np.mean(fscore_history)


def SolverFactory(config, dataset_name, train_loader, val_loader, test_loader):
    dataset_name = dataset_name.lower()
    if dataset_name == "s_mrhisum":
        return MrHiSumSolver(config, train_loader, val_loader, test_loader)
    elif dataset_name == "s_videoxum":
        return VideoXumSolver(config, train_loader, val_loader, test_loader)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
