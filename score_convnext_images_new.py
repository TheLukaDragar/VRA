print("hello")
import os
import warnings

print("importing modules")
from dataset_tool import (
    RandomSeqFaceFramesDataset,
    FaceFramesSeqPredictionDataset_all_frames,
)
from dataset_tool import build_transforms

print("imported dataset_1")
import torch
import torch.nn as nn

print("imported torch")
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

print("imported pytorch_lightning")
# WandbLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

print("imported pytorch_lightning callbacks")
import argparse

from scipy.stats import pearsonr, spearmanr

from timm import create_model

print("imported timm")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import wandb
import random
import numpy as np
from lightning.pytorch import seed_everything

# tqdm
from tqdm import tqdm


def train_val_test_split(dataset, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    assert (
        0 <= train_prop <= 1 and 0 <= val_prop <= 1 and 0 <= test_prop <= 1
    ), "Proportions must be between 0 and 1"
    assert (
        round(train_prop + val_prop + test_prop, 10) == 1
    ), "Proportions must sum to 1"

    total_length = len(dataset)
    train_length = int(train_prop * total_length)
    val_length = int(val_prop * total_length)
    test_length = total_length - train_length - val_length

    return random_split(dataset, [train_length, val_length, test_length])


class ConvNeXt(pl.LightningModule):
    def __init__(self, og_path, model_name="convnext_tiny", dropout=0.1, loss="rmse"):
        super(ConvNeXt, self).__init__()
        self.model_name = model_name
        self.backbone = create_model(self.model_name, pretrained=True, num_classes=2)
        # load from checkpoint
        # self.backbone.load_state_dict(torch.load('/d/hpc/projects/FRI/ldragar/convnext_xlarge_384_in22ft1k_10.pth'))

        n_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Linear(n_features, 2)
        self.backbone = torch.nn.DataParallel(self.backbone)
        self.backbone.load_state_dict(torch.load(og_path))

        self.backbone = self.backbone.module
        self.backbone.head.fc = nn.Identity()

        self.drop = nn.Dropout(dropout)

        # one frame feature vector
        self.fc = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 1)
        # self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, 1)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.test_preds = {}
        self.test_labels = {}
        self.current_test_set = None

        # Initialize PearsonCorrCoef metric
        self.pearson_corr_coef = torchmetrics.PearsonCorrCoef().to(self.device)
        self.spearman_corr_coef = torchmetrics.SpearmanCorrCoef().to(self.device)
        # rmse
        self.mse_log = torchmetrics.MeanSquaredError().to(self.device)

        # ddp debug
        self.seen_samples = set()
        self.seen_sample_names = set()

        self.loss_fn = None

        if loss == "rmse":
            self.loss_fn = self.RMSE
        elif loss == "mae":
            self.loss_fn = self.MAE

        elif loss == "opdai":

            def custom_loss(pred, target):
                return self.norm_loss_with_normalization(
                    pred, target, p=1, q=2
                ) + self.kl_div_loss(pred, target)

            self.loss_fn = custom_loss

        elif loss == "hust":

            def custom_loss(pred, target):
                # combine mae with rank and pearson
                alpha = 0.5
                beta = 1
                # L = LMAE + α · LP LCC + β · Lrank (
                return (
                    self.MAE(pred, target)
                    + alpha * self.pearson_corr_coef_loss(pred, target)
                    # + beta * self.pairwise_ranking_loss(pred, target) #TODO
                )

            self.loss_fn = custom_loss
        else:
            raise ValueError("Invalid loss function")

        self.save_hyperparameters()

    def RMSE(self, preds, y):
        mse = self.mse(preds.view(-1), y.view(-1))
        return torch.sqrt(mse)

    def forward(self, x):
        # Choose a random index from the sequence dimension
        random_idx = torch.randint(0, x.shape[1], (x.shape[0],))

        # Select a random frame for each item in the batch
        x_random_frame = x[torch.arange(x.shape[0]), random_idx]

        # Process the selected frame with the backbone
        features = self.backbone(
            x_random_frame
        )  # Output shape: (batch_size, n_features)

        # Optionally, you can continue with further processing as needed
        x = self.drop(features)
        x = torch.nn.functional.relu(self.fc(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc2(x)
        logit = x

        return logit

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        rmse_loss = self.RMSE(preds, y)
        self.log(
            "train_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train_rmse_loss",
            rmse_loss.item(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        rmse_loss = self.RMSE(preds, y)
        loss_value = loss.item()

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_rmse_loss",
            rmse_loss.item(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # if loss is nan.0 then stop training
        if math.isnan(loss_value):
            print("val_loss is nan.0 stopping training")
            self.trainer.should_stop = True
        elif math.isinf(loss_value):
            print("val_loss is inf stopping training")
            self.trainer.should_stop = True

    def pearson_corr_coef_mine(self, preds, y, eps=1e-8):
        preds_mean = preds.mean()
        y_mean = y.mean()
        cov = ((preds - preds_mean) * (y - y_mean)).mean()
        preds_std = torch.sqrt((preds - preds_mean).pow(2).mean() + eps)
        y_std = torch.sqrt((y - y_mean).pow(2).mean() + eps)
        plcc = cov / (preds_std * y_std)
        return plcc

    def pearson_corr_coef_loss(self, preds, y):
        plcc = self.pearson_corr_coef_mine(preds.view(-1), y.view(-1))
        loss = 1 - torch.abs(plcc)
        return loss

    def kl_div_loss(self, preds, target, eps=1e-8):
        preds_softmax = torch.nn.functional.softmax(preds, dim=-1)
        target_softmax = torch.nn.functional.softmax(target, dim=-1)
        loss = torch.sum(
            target_softmax * torch.log((target_softmax + eps) / (preds_softmax + eps)),
            dim=-1,
        )
        loss = torch.mean(loss)
        return loss

    def norm_loss_with_normalization(self, pred, target, p=1, q=2):
        """
        Args:
            pred (Tensor): of shape (N, 1). Predicted tensor.
            target (Tensor): of shape (N, 1). Ground truth tensor.
        """
        batch_size = pred.shape[0]
        if batch_size > 1:
            vx = pred - pred.mean()
            vy = target - target.mean()
            if torch.all(vx == 0) or torch.all(vy == 0):
                return torch.nn.functional.l1_loss(pred, target)

            scale = np.power(2, p) * np.power(batch_size, max(0, 1 - p / q))  # p, q>0
            norm_pred = torch.nn.functional.normalize(vx, p=q, dim=0)
            norm_target = torch.nn.functional.normalize(vy, p=q, dim=0)
            loss = torch.norm(norm_pred - norm_target, p=p) / scale
        else:
            loss = torch.nn.functional.l1_loss(pred, target)
        return loss.mean()

    def MAE(self, preds, y):
        return self.mae(preds.view(-1), y.view(-1))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def test_step(self, batch, batch_idx):
        x, y, name = batch
        # Check for duplicates THIS IS FOR DDP DEBUGGING
        # for sample in x:
        #     sample_hash = hash(sample.cpu().numpy().tostring())
        #     if sample_hash in self.seen_samples:
        #         print("Duplicate sample detected:", sample)
        #         warnings.warn("Duplicate sample detected!!! ")
        #     else:
        #         # print("New sample detected:", str(sample_hash),str(batch_idx))
        #         self.seen_samples.add(sample_hash)

        for sample_name in name:
            if sample_name in self.seen_sample_names:
                print("Duplicate sample name detected:", sample_name)
                warnings.warn("Duplicate sample name detected!!! ")
            else:
                # print("New sample name detected:", sample_name)
                self.seen_sample_names.add(sample_name)
                # print("New sample name detected:", sample_name)

        preds = self(x)

        if self.current_test_set not in self.test_preds:
            self.test_preds[self.current_test_set] = []

        self.test_preds[self.current_test_set].append(preds)

        if self.current_test_set not in self.test_labels:
            self.test_labels[self.current_test_set] = []

        self.test_labels[self.current_test_set].append(y)

    def on_test_epoch_end(self):
        test_preds = torch.cat(self.test_preds[self.current_test_set])
        test_labels = torch.cat(self.test_labels[self.current_test_set])
        test_preds = test_preds.view(-1)
        plcc = self.pearson_corr_coef(test_preds, test_labels)
        spearman = self.spearman_corr_coef(test_preds, test_labels)
        mse_log = self.mse_log(test_preds, test_labels)
        rmse = torch.sqrt(mse_log)

        self.log(self.current_test_set + "_plcc", plcc, sync_dist=True)
        self.log(self.current_test_set + "_spearman", spearman, sync_dist=True)
        self.log(
            self.current_test_set + "_score", (plcc + spearman) / 2, sync_dist=True
        )
        self.log(self.current_test_set + "_rmse", rmse, sync_dist=True)

    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with the given parameters."
    )

    parser.add_argument(
        "--dataset_root",
        default="../dataset",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--labels_dir", default="./label/", help="Path to the labels directory."
    )
    # parser.add_argument('--cp_save_dir', default='/d/hpc/projects/FRI/ldragar/checkpoints/', help='Path to save checkpoints.')
    parser.add_argument(
        "--model_dir",
        default="./convnext_models_images/",
        help="Path to save the final model.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length.")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. for reproducibility. -1 for no seed.",
    )
    parser.add_argument(
        "--cp_id",
        default="37orwro0",
        help="id(wandb_id) of the checkpoint to load from the model_dir.",
    )
    parser.add_argument(
        "--x_predictions",
        type=int,
        default=1,
        help="Number of predictions to make. then average them.",
    )
    parser.add_argument(
        "--out_predictions_dir",
        default="./predictions/",
        help="Path to save the predictions.",
    )
    parser.add_argument(
        "--test_labels_dir",
        default="./competition_end_groundtruth/",
        help="Path to the test labels directory.",
    )
    parser.add_argument(
        "--og_checkpoint",
        default="./DFGC-1st-2022-model/convnext_xlarge_384_in22ft1k_30.pth",
        help="DFGC1st convnext_xlarge_384_in22ft1k_30.pth file path",
    )
    # stage array
    parser.add_argument(
        "--max_frames",
        type=int,
        default=250,
        help="max frames to use for each video",
    )

    args = parser.parse_args()

    print("starting")
    dataset_root = args.dataset_root
    # labels_dir = args.labels_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    seq_len = args.seq_len
    seed = args.seed
    cp_id = args.cp_id
    x_predictions = args.x_predictions
    out_predictions_dir = args.out_predictions_dir
    og_path = args.og_checkpoint

    # cp_save_dir = args.cp_save_dir
    test_labels_dir = args.test_labels_dir

    if not seed == -1:
        seed_everything(seed, workers=True)

    transform_train, transform_test = build_transforms(
        384,
        384,
        max_pixel_value=255.0,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )
    _, transform_test_LR = build_transforms(
        384,
        384,
        max_pixel_value=255.0,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        test_lr_flip=True,
    )

    model = ConvNeXt(
        og_path, model_name="convnext_xlarge_384_in22ft1k", dropout=0.1, loss="rmse"
    )
    # load checkpoint
    # get files in dir
    files = os.listdir(model_dir)
    # get the one with the same run id
    cp_name = [f for f in files if cp_id in f][0]

    print(f"loading model from checkpoint {cp_name}")
    if not cp_name.endswith(".ckpt"):
        # this is a pt file
        cp_name = os.path.join(model_dir, cp_name, cp_name + ".pt")

    # load checkpoint
    if cp_name.endswith(".ckpt"):
        # load checkpoint
        checkpoint = torch.load(os.path.join(model_dir, cp_name))
        model.load_state_dict(checkpoint["state_dict"])

    else:
        # load pt file
        model.load_state_dict(torch.load(cp_name))

    # PREDICTIONS
    model.eval()  # set the model to evaluation mode

    model = model.to("cuda:0")

    stages = ["1", "2", "3"]

    resultsdir = os.path.join(out_predictions_dir, cp_id, str(seed))
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir, exist_ok=True)

    class Result:
        def __init__(self, test1, test2, test3, name, fn1, fn2, fn3):
            self.test1 = test1
            self.test2 = test2
            self.test3 = test3
            self.name = name
            self.fn1 = fn1
            self.fn2 = fn2
            self.fn3 = fn3
            self.summary = None
            self.weight = None

        def set_summary(self, summary):
            self.summary = summary

        def set_weight(self, weight):
            self.weight = weight

    def score_model(predictions, groundtruth):
        pearson_corr_coef = pearsonr(predictions, groundtruth)[0]
        spearman_corr_coef = spearmanr(predictions, groundtruth)[0]
        mse = np.mean((predictions - groundtruth) ** 2)
        rmse = np.sqrt(mse)
        return {
            "pearson_corr_coef": pearson_corr_coef,
            "spearman_corr_coef": spearman_corr_coef,
            "rmse": rmse,
        }

    def final_score(predictions, groundtruth):
        scores = []
        for i in range(3):
            scores.append(score_model(predictions[i], groundtruth[i]))
        return (
            scores[0]["pearson_corr_coef"]
            + scores[0]["spearman_corr_coef"]
            + scores[1]["pearson_corr_coef"]
            + scores[1]["spearman_corr_coef"]
            + scores[2]["pearson_corr_coef"]
            + scores[2]["spearman_corr_coef"]
        ) / 6

    t1 = []
    t1_lr = []
    t1_non_flip = []
    t2 = []
    t2_lr = []
    t2_non_flip = []
    t3 = []
    t3_lr = []
    t3_non_flip = []

    gt1 = []
    gt2 = []
    gt3 = []

    for stage in stages:
        name = "Test" + stage + "-labels.txt"

        # Initialize lists to store the predictions and the test names
        all_test_labels = []
        all_test_labels_lr = []
        all_test_names = []
        all_test_gt = []
        all_test_std = []
        all_test_std_lr = []
        min_test_frames_scores = []

        # # Make x_predictions for each stage
        # for i in range(x_predictions):
        #     test_labels = []
        #     test_names = []
        #     test_gt = []
        #     test_std = []
        #     test_frames_scores = []

        # ds = FaceFramesSeqPredictionDataset(
        #     os.path.join(labels_dir, name),
        #     dataset_root,
        #     transform=transform_test,
        #     seq_len=seq_len,
        # )
        ds = FaceFramesSeqPredictionDataset_all_frames(
            os.path.join(test_labels_dir, name),
            dataset_root,
            transform=transform_test,
            max_frames=args.max_frames,
        )
        ds_lr = FaceFramesSeqPredictionDataset_all_frames(
            os.path.join(test_labels_dir, name),
            dataset_root,
            transform=transform_test_LR,
            max_frames=args.max_frames,
        )

        print(f"loaded {len(ds)} test examples")

        # with torch.no_grad():
        #     for x, gt, nameee in ds:
        #         x = x.unsqueeze(0).to(model.device)
        #         y = model(x)
        #         y = y.cpu().numpy()
        #         y = y[0][0]

        #         test_labels.append(y)
        #         test_names.append(nameee)
        #         test_gt.append(gt)

        # dataloader
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        dl_lr = DataLoader(
            ds_lr,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        with torch.no_grad():
            # use tqdm

            with open(
                os.path.join(
                    resultsdir, "frame_predictions_Test" + stage + "_preds.txt"
                ),
                "w",
            ) as f:
                for (sequences, gt, name), (sequences_lr, _, _) in tqdm(
                    zip(dl, dl_lr), desc=f"Predicting {name}", total=len(dl)
                ):
                    print(f"predicting {name}")
                    # print("sequences", sequences.shape)

                    # sequences_lr, _, _ = next(iter(dl_lr))

                    predictions = []

                    # make predictions for each frame in the video
                    # make a batch of size number of frames in the video

                    sequences = sequences.permute(1, 0, 2, 3, 4)
                    sequences_lr = sequences_lr.permute(1, 0, 2, 3, 4)

                    # sequences_lr = torch.flip(sequences, [3])

                    print("sequences", sequences.shape)

                    sequences = sequences.to(model.device)
                    y = model(sequences)
                    y = y.cpu().numpy()
                    torch.cuda.empty_cache()  # Clear GPU cache

                    sequences_lr = sequences_lr.to(model.device)
                    y_lr = model(sequences_lr)
                    y_lr = y_lr.cpu().numpy()
                    torch.cuda.empty_cache()  # Clear GPU cache

                    # print("y", y)
                    # print("y shape", y.shape)
                    # remove batch dim

                    y = y.squeeze(1)
                    y_lr = y_lr.squeeze(1)

                    # print(
                    #     "y", y
                    # )  # y is now a list of predictions for each frame in the video

                    predictions = y
                    predictions_lr = y_lr

                    # print(predictions.shape)
                    # Perform prediction on each frame in the video
                    # for frame in sequences:
                    #     print("frame", frame.shape)
                    #     frame = frame.to(model.device)
                    #     y = model(frame)
                    #     y = y.cpu().numpy()
                    #     print("y", y)
                    #     print("y shape", y.shape)

                    #     y = y[0][0]
                    #     predictions.append(y)

                    # Compute mean and standard deviation of predictions
                    mean_prediction = np.mean(predictions)
                    std_prediction = np.std(predictions)
                    mean_prediction_lr = np.mean(predictions_lr)
                    std_prediction_lr = np.std(predictions_lr)

                    # print("score:",mean_prediction)
                    # print("std:",std_prediction)
                    # test_labels.append(mean_prediction)
                    # test_names.append(name)
                    # test_gt.append(gt)
                    # test_std.append(std_prediction)
                    # # test_frames_scores.append(predictions)
                    # min_test_frames_scores.append(min(predictions))

                    all_test_labels.append(mean_prediction.item())
                    all_test_labels_lr.append(mean_prediction_lr.item())
                    all_test_names.append(name[0])
                    all_test_gt.append(gt.item())
                    all_test_std.append(std_prediction.item())
                    all_test_std_lr.append(std_prediction_lr.item())
                    min_test_frames_scores.append(min(predictions))

                    f.write(f"{name[0]},{ ','.join([str(x) for x in predictions]) }\n")

            print(f"predicted {len(all_test_labels)} labels for {name}")

        # # Calculate the mean and standard deviation of the predictions
        # all_test_labels = np.array(all_test_labels)
        mean_test_labels = all_test_labels
        mean_test_labels_lr = all_test_labels_lr
        # mean_test_labels = np.mean(all_test_labels, axis=0)
        # std_test_labels = np.std(all_test_labels, axis=0)
        # std_beetwen_frames = np.mean(all_test_std, axis=0)
        # check if all_test_gt has the same values in all the positions
        # if not np.all(all_test_gt == all_test_gt[0]):
        #     print("all_test_gt has different values in different positions")

        print(f"all_test_gt: {all_test_gt}")
        mean_test_gt = all_test_gt

        # Calculate the RMSE between each pair of predictions
        # rmse_list = []
        # for i in range(x_predictions):
        #     for j in range(i + 1, x_predictions):
        #         rmse = np.sqrt(np.mean((all_test_labels[i] - all_test_labels[j]) ** 2))
        #         rmse_list.append(rmse)

        # print(f"RMSE between predictions: {rmse_list}")

        # Save the mean predictions to a file
        with open(
            os.path.join(resultsdir, "Test" + stage + "non_flip_preds.txt"), "w"
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{mean_test_labels[i]}\n")

        with open(
            os.path.join(resultsdir, "Test" + stage + "flip_preds.txt"), "w"
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{mean_test_labels_lr[i]}\n")

        # combine predictions
        mean_test_labels2 = []
        for i in range(len(all_test_names)):
            mean_test_labels2.append((mean_test_labels[i] + mean_test_labels_lr[i]) / 2)

        with open(os.path.join(resultsdir, "Test" + stage + "_preds.txt"), "w") as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{mean_test_labels2[i]}\n")

        # save std beetwen frames
        with open(
            os.path.join(resultsdir, "std_beetwen_frames_Test" + stage + "_preds.txt"),
            "w",
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{all_test_std[i]}\n")

        with open(
            os.path.join(
                resultsdir, "std_beetwen_frames_Test" + stage + "_preds_lr.txt"
            ),
            "w",
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{all_test_std_lr[i]}\n")

        # # save all predictions for each frame
        # with open(
        #     os.path.join(resultsdir, "frame_predictions+Test" + stage + "_preds.txt"),
        #     "w",
        # ) as f:
        #     for i in range(len(all_test_frames_scores)):
        #         line = ""
        #         for j in range(len(all_test_frames_scores[i])):
        #             line += str(all_test_frames_scores[i][j]) + ","

        #         f.write(f"{line}\n")

        # save lowest
        with open(
            os.path.join(resultsdir, "lowest_Test" + stage + "_preds.txt"), "w"
        ) as f:
            for i in range(len(all_test_names)):
                f.write(f"{all_test_names[i]},{min_test_frames_scores[i]}\n")

        if stage == "1":
            t1 = np.array(mean_test_labels2)
            t1_lr = np.array(mean_test_labels_lr)
            t1_non_flip = np.array(mean_test_labels)
            gt1 = np.array(mean_test_gt)

        if stage == "2":
            t2 = np.array(mean_test_labels2)
            t2_lr = np.array(mean_test_labels_lr)
            t2_non_flip = np.array(mean_test_labels)
            gt2 = np.array(mean_test_gt)

        if stage == "3":
            t3 = np.array(mean_test_labels2)
            t3_lr = np.array(mean_test_labels_lr)
            t3_non_flip = np.array(mean_test_labels)
            gt3 = np.array(mean_test_gt)

        print(
            f"saved {len(mean_test_labels)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds.txt')}"
        )

    # score model
    print("scoring model")
    test_set1_score = score_model(t1, gt1)
    test_set2_score = score_model(t2, gt2)
    test_set3_score = score_model(t3, gt3)
    final_score = final_score([t1, t2, t3], [gt1, gt2, gt3])
    print("test_set1_score", test_set1_score)
    print("test_set2_score", test_set2_score)
    print("test_set3_score", test_set3_score)
    print("final_score", final_score)

    # score with non flip
    test_set1_score_non_flip = score_model(t1_non_flip, gt1)
    test_set2_score_non_flip = score_model(t2_non_flip, gt2)
    test_set3_score_non_flip = score_model(t3_non_flip, gt3)
    final_score_non_flip = final_score(
        [t1_non_flip, t2_non_flip, t3_non_flip], [gt1, gt2, gt3]
    )
    print("test_set1_score_non_flip", test_set1_score_non_flip)
    print("test_set2_score_non_flip", test_set2_score_non_flip)
    print("test_set3_score_non_flip", test_set3_score_non_flip)

    print("final_score_non_flip", final_score_non_flip)

    # score with lr flip
    test_set1_score_lr = score_model(t1_lr, gt1)
    test_set2_score_lr = score_model(t2_lr, gt2)
    test_set3_score_lr = score_model(t3_lr, gt3)
    final_score_lr = final_score([t1_lr, t2_lr, t3_lr], [gt1, gt2, gt3])
    print("test_set1_score_lr", test_set1_score_lr)
    print("test_set2_score_lr", test_set2_score_lr)
    print("test_set3_score_lr", test_set3_score_lr)
    print("final_score_lr", final_score_lr)

    # save to resultsdir
    with open(os.path.join(resultsdir, "scores.txt"), "w") as f:
        f.write(f"cp_id,{cp_id}\n")
        f.write(f"seed,{seed}\n")
        f.write(f"args,{args}\n")
        f.write(f"test_set1_score,{test_set1_score}\n")
        f.write(f"test_set2_score,{test_set2_score}\n")
        f.write(f"test_set3_score,{test_set3_score}\n")

        f.write(f"test_set1_score_non_flip,{test_set1_score_non_flip}\n")
        f.write(f"test_set2_score_non_flip,{test_set2_score_non_flip}\n")
        f.write(f"test_set3_score_non_flip,{test_set3_score_non_flip}\n")

        f.write(f"test_set1_score_lr,{test_set1_score_lr}\n")
        f.write(f"test_set2_score_lr,{test_set2_score_lr}\n")
        f.write(f"test_set3_score_lr,{test_set3_score_lr}\n")
        f.write(f"final_score_lr,{final_score_lr}\n")
        f.write(f"final_score_non_flip,{final_score_non_flip}\n")
        f.write(f"final_score,{final_score}\n")

    # save std beetwen frames
    # with open(os.path.join(resultsdir, "std_beetwen_frames.txt"), "w") as f:
    #     for i in range(len(std_beetwen_frames)):
    #         f.write(f"{std_beetwen_frames[i]}\n")

    # # compute mae beetwen submitet result

    # # read all files in the folder
    # submition = "./eva_models/37orwro0/dotren_ema37orwro0_final_modl_10_avgpreds/"
    # names = ["Test1_preds.txt", "Test2_preds.txt", "Test3_preds.txt"]
    # results = []

    # test1 = []
    # test2 = []
    # test3 = []
    # fn1 = []
    # fn2 = []
    # fn3 = []
    # with open(os.path.join(submition, names[0])) as f1:
    #     for line in f1:
    #         # C1/3-1-2-submit-00000.mp4,2.9382266998291016
    #         l = line.split(",")
    #         fn1.append(l[0])
    #         test1.append(float(l[1]))

    # with open(os.path.join(submition, names[1])) as f2:
    #     for line in f2:
    #         # C1/3-1-2-submit-00000.mp4,2.9382266998291016

    #         l = line.split(",")
    #         fn2.append(l[0])
    #         test2.append(float(l[1]))

    # with open(os.path.join(submition, names[2])) as f3:
    #     for line in f3:
    #         # C1/3-1-2-submit-00000.mp4,2.9382266998291016

    #         l = line.split(",")
    #         fn3.append(l[0])
    #         test3.append(float(l[1]))

    # submition = Result(
    #     test1, test2, test3, "dotren_ema37orwro0_final_modl_10_avgpreds", fn1, fn2, fn3
    # )
    # new_result = Result(t1, t2, t3, resultsdir, fn1, fn2, fn3)

    # from sklearn.metrics import mean_absolute_error

    # # compute mae
    # mae1 = mean_absolute_error(submition.test1, new_result.test1)
    # mae2 = mean_absolute_error(submition.test2, new_result.test2)
    # mae3 = mean_absolute_error(submition.test3, new_result.test3)
    # mae = (mae1 + mae2 + mae3) / 3
    # print("mae1", mae1)
    # print("mae2", mae2)
    # print("mae3", mae3)
    # print("avg_mae", mae)
    # # save to resultsdir
    # with open(os.path.join(resultsdir, "mae.txt"), "w") as f:
    #     f.write(f"mae1,{mae1}\n")
    #     f.write(f"mae2,{mae2}\n")
    #     f.write(f"mae3,{mae3}\n")
    #     f.write(f"avg_mae,{mae}\n")
    #     f.write(f"new_result,{new_result.name}\n")
    #     f.write(f"seed,{seed}\n")
