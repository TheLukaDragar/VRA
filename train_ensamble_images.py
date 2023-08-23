print("hello")
import os
import warnings

print("importing modules")
from dataset_tool import RandomSeqFaceFramesDataset, FaceFramesSeqPredictionDataset_middle_frames
from dataset_tool import build_transforms
import math

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
import random
import numpy as np
from lightning.pytorch import seed_everything


def train_val_split(dataset, train_prop=0.8, val_prop=0.2, seed=None):
    assert (
        0 <= train_prop <= 1 and 0 <= val_prop <= 1
    ), "Proportions must be between 0 and 1"
    assert round(train_prop + val_prop, 10) == 1, "Proportions must sum to 1"

    total_length = len(dataset)
    train_length = int(train_prop * total_length)
    val_length = int(val_prop * total_length)

    if seed is not None:
        return random_split(
            dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        return random_split(dataset, [train_length, val_length])


class Ensamble(pl.LightningModule):
    def __init__(self, og_path, model_name="convnext_tiny",  model_name2="eva_large_patch14_336.in22k_ft_in22k_in1k", loss="rmse", lr=2e-5):
        super(Ensamble, self).__init__()
        self.model_name = model_name + model_name2
       
        self.backbone = create_model(model_name, pretrained=True, num_classes=2)
        # load from checkpoint
        # self.backbone.load_state_dict(torch.load('/d/hpc/projects/FRI/ldragar/convnext_xlarge_384_in22ft1k_10.pth'))

        n_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Linear(n_features, 2)
        self.backbone = torch.nn.DataParallel(self.backbone)
        self.backbone.load_state_dict(torch.load(og_path))

        self.backbone = self.backbone.module
        self.backbone.head.fc = nn.Identity()

        self.backbone2 = create_model(model_name2, pretrained=True, num_classes=0)

        n_features2 = 1024


        self.drop = nn.Dropout(dropout)

        # one frame feature vector
        self.fc = nn.Linear(n_features + n_features2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
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
        self.lr = lr

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
        features1 = self.backbone(x_random_frame)  # Output shape: (batch_size, n_features)


        #e x from 384 to 336 to match the other model
        x_random_frame_resized = torch.nn.functional.interpolate(x_random_frame, size=(336, 336), mode='bilinear', align_corners=False)


        features2 = self.backbone2(x_random_frame_resized)  # Output shape: (batch_size, n_features)

        # Concatenate the features
        features = torch.cat((features1, features2), dim=1)  # Output shape: (batch_size, size_of_features1 + size_of_features2)



        # Optionally, you can continue with further processing as needed
        x = self.drop(features)
        x = torch.nn.functional.relu(self.fc(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = self.fc6(x)
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

        #if loss is nan.0 then stop training
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
            target_softmax * torch.log((target_softmax + eps) / (preds_softmax + eps)), dim=-1
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
        optimizer = AdamW(self.parameters(), lr=self.lr)
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
        "--labels_file",
        default="./label/train_set.csv",
        help="Path to the labels train file.",
    )
    # test_labels_dir
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
    # parser.add_argument('--cp_save_dir', default='/d/hpc/projects/FRI/ldragar/checkpoints/', help='Path to save checkpoints.')
    parser.add_argument(
        "--final_model_save_dir",
        default="./ensemble_models_images/",
        help="Path to save the final model.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length.")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. for reproducibility. Note final model is worse with seed set.",
    )
    parser.add_argument(
        "--wdb_project_name",
        default="luka_vra",
        help="Weights and Biases project name.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradients over n batches.",
    )
    # experiment_name
    parser.add_argument(
        "--experiment_name",
        default="ensemble_images",
        help="Experiment name.",
    )
    parser.add_argument(
        "--wandb_resume_version", default="None", help="Wandb resume version."
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes.")
    # devices array
    parser.add_argument(
        "--devices", nargs="+", type=int, default=[0, 1], help="Devices to train on."
    )
    # drop out
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    # max_epochs
    parser.add_argument("--max_epochs", type=int, default=33, help="Max epochs.")

    parser.add_argument(
        "--loss",
        default="rmse",
        help="Loss function to use. Supported values: mse, rmse, mae, norm_loss_with_normalization, opdai, hust",
    )
    #lr
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")

    parser.add_argument(
        "--augmentation",
        type=bool,
        default=False,
        help="Whether to use data augmentation.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of the dataset to use for validation.",
    )

    parser.add_argument(
        "--from_cp_id",
        default="None",
        help="Resume training from checkpoint id.",
    )


    # parser.add_argument('--test_labels_dir', default='/d/hpc/projects/FRI/ldragar/label/', help='Path to the test labels directory.')

    args = parser.parse_args()

    print("starting")
    dataset_root = args.dataset_root
    labels_file = args.labels_file
    og_path = args.og_checkpoint
    final_model_save_dir = args.final_model_save_dir
    batch_size = args.batch_size
    seq_len = args.seq_len
    seed = args.seed
    wdb_project_name = args.wdb_project_name
    accumulate_grad_batches = args.accumulate_grad_batches
    max_epochs = args.max_epochs

    # cp_save_dir = args.cp_save_dir
    # test_labels_dir = args.test_labels_dir

    if seed != -1:
        seed_everything(seed, workers=True)

    # seed only data

    transform_train, transform_test = build_transforms(
        384,
        384,
        max_pixel_value=255.0,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        augment=args.augmentation,
    )

    print("loading dataset")

    face_frames_dataset = RandomSeqFaceFramesDataset(
        dataset_root, labels_file, transform=transform_train, seq_len=seq_len,seed= seed if seed != -1 else None
    )

    face_frames_dataset_test1 = FaceFramesSeqPredictionDataset_middle_frames(
        os.path.join(args.test_labels_dir, "Test1-labels.txt"),
        dataset_root,
        transform=transform_test,
        seq_len=1,
        seed = seed if seed != -1 else None
    )
    face_frames_dataset_test2 = FaceFramesSeqPredictionDataset_middle_frames(
        os.path.join(args.test_labels_dir, "Test2-labels.txt"),
        dataset_root,
        transform=transform_test,
        seq_len=1,
        seed= seed if seed != -1 else None
    )
    face_frames_dataset_test3 = FaceFramesSeqPredictionDataset_middle_frames(
        os.path.join(args.test_labels_dir, "Test3-labels.txt"),
        dataset_root,
        transform=transform_test,
        seq_len=1,
       seed= seed if seed != -1 else None
    )

    print("splitting dataset")
    train_ds, val_ds = train_val_split(face_frames_dataset,train_prop=(1-args.val_split),val_prop=args.val_split,seed=seed if seed != -1 else None)

    print("first 5 train labels")
    for i in range(5):
        print(train_ds[i][1])
    print("first 5 val labels")
    for i in range(5):
        print(val_ds[i][1])
    print("first 5 test1 labels")
    for i in range(5):
        print(face_frames_dataset_test1[i][1])
    print("first 5 test2 labels")
    for i in range(5):
        print(face_frames_dataset_test2[i][1])
    print("first 5 test3 labels")
    for i in range(5):
        print(face_frames_dataset_test3[i][1])

    print("loading dataloader")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    test_dl1 = DataLoader(
        face_frames_dataset_test1, batch_size=1, shuffle=False, num_workers=4
    )
    test_dl2 = DataLoader(
        face_frames_dataset_test2, batch_size=1, shuffle=False, num_workers=4
    )
    test_dl3 = DataLoader(
        face_frames_dataset_test3, batch_size=1, shuffle=False, num_workers=4
    )

    # if len(test_dl1) % batch_size != 0 or len(test_dl2) % batch_size != 0 or len(test_dl3) % batch_size != 0:
    #     warnings.warn("Uneven inputs detected! With multi-device settings, DistributedSampler may replicate some samples to ensure all devices have the same batch size. TEST WILL NOT BE ACCURATE CRITICAL!")
    #     exit(1)

    print(
        "train dataset will be split into",
        len(train_dl),
        "batches",
        "each batch will have",
        batch_size,
        "samples",
    )
    print(
        "val dataset will be split into",
        len(val_dl),
        "batches",
        "each batch will have",
        batch_size,
        "samples",
    )
    print(
        "each gpu will process batch size of",
        batch_size,
        "samples",
        "global batch size will be",
        batch_size * len(args.devices),
    )

    if len(train_dl) % batch_size != 0 or len(val_dl) % batch_size != 0:
        warnings.warn(
            "Uneven inputs detected! With multi-device settings, DistributedSampler may replicate some samples to ensure all devices have the same batch size."
        )

    print(f"loaded {len(train_dl)} train batches and {len(val_dl)} val batches")
    print(
        f"loaded {len(test_dl1)} test batches for test set 1 of {len(face_frames_dataset_test1)} examples"
    )
    print(
        f"loaded {len(test_dl2)} test batches for test set 2 of {len(face_frames_dataset_test2)} examples"
    )
    print(
        f"loaded {len(test_dl3)} test batches for test set 3 of {len(face_frames_dataset_test3)} examples"
    )

    # print first train example

    for x, y in train_dl:
        print(x.shape)
        print(y.shape)
        print(y)
        break

    if args.wandb_resume_version == "None":
        wandb_logger = WandbLogger(name=args.experiment_name, project=wdb_project_name)
    else:
        wandb_logger = WandbLogger(
            name=args.experiment_name, version=args.wandb_resume_version, resume="must"
        )

    # convnext_xlarge_384_in22ft1k
    model = Ensamble(
        og_path, model_name="convnext_xlarge_384_in22ft1k",model_name2="eva_large_patch14_336.in22k_ft_in22k_in1k",
         dropout=args.dropout, loss=args.loss,lr=args.lr
    )

    if args.from_cp_id != "None":
        checkpoint_path = os.path.join(final_model_save_dir,args.from_cp_id,f"{args.from_cp_id}.pt")
        print(f"loading model from checkpoint {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

    wandb_logger.watch(model, log="all", log_freq=100)
    # log batch size
    wandb_logger.log_hyperparams({"batch_size": batch_size})
    # random face frames
    wandb_logger.log_hyperparams({"seq_len": seq_len})
    # log seed
    wandb_logger.log_hyperparams({"seed": seed})
    # log accumulate_grad_batches
    wandb_logger.log_hyperparams({"accumulate_grad_batches": accumulate_grad_batches})
    # devices
    wandb_logger.log_hyperparams({"devices": args.devices})

    wandb_logger.log_hyperparams({"augment": args.augmentation})

    wandb_logger.log_hyperparams({"val_split": args.val_split})


    wandb_run_id = str(wandb_logger.version)
    if wandb_run_id == "None":
        print("no wandb run id this is a copy of model with DDP")

    print("init trainer")

    # save hyperparameters
    wandb_logger.log_hyperparams(model.hparams)

    # checkpoint_callback = ModelCheckpoint(monitor='val_loss',
    #                                     dirpath=cp_save_dir,
    #                                     filename=f'{wandb_run_id}-{{epoch:02d}}-{{val_loss:.2f}}', mode='min', save_top_k=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_epochs=max_epochs,
        log_every_n_steps=200,
        callbacks=[
            # EarlyStopping(monitor="val_loss",
            #             mode="min",
            #             patience=4,
            #             ),
            # checkpoint_callback
        ],
        logger=wandb_logger,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=seed != -1,
    )

    print("start training")
    # Train the model
    trainer.fit(model, train_dl, val_dl)
    # Test the model
    print("testing current model")

    # model.current_test_set = "test_set1"
    # trainer.test(model, test_dl1)
    # model.current_test_set = "test_set2"
    # trainer.test(model, test_dl2)
    # model.current_test_set = "test_set3"
    # trainer.test(model, test_dl3)

    if trainer.global_rank == 0:
        # test_set1_score = wandb_logger.experiment.summary["test_set1_score"]
        # test_set2_score = wandb_logger.experiment.summary["test_set2_score"]
        # test_set3_score = wandb_logger.experiment.summary["test_set3_score"]
        # avg_score = (test_set1_score + test_set2_score + test_set3_score) / 3
        # print(f"test_set1_score: {test_set1_score}")
        # print(f"test_set2_score: {test_set2_score}")
        # print(f"test_set3_score: {test_set3_score}")
        # print(f"final_score: {avg_score}")

        # # Log the average score to WandB
        # wandb.log({"final_score": avg_score})
        # save model
        model_path = os.path.join(final_model_save_dir, wandb_run_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path, f"{wandb_run_id}.pt"))
        print(f"finished training, saved model to {model_path}")

        #PREDICTIONS
        model.eval()  # set the model to evaluation mode

        model = model.to('cuda:0')

        stages = ['1','2','3']
        #get wandb run id

        resultsdir = os.path.join('./results/', wandb_run_id)
        if not os.path.exists(resultsdir):
            os.makedirs(resultsdir)


        scores=[]

        for stage in stages:
            name='test_set'+stage+'.txt'
            test_labels = []
            test_names = []
            test_gt = []

            #use seq len

            ds = FaceFramesSeqPredictionDataset_middle_frames(
                os.path.join(args.test_labels_dir, "Test"+stage+"-labels.txt"),
                dataset_root,
                transform=transform_test,
                seq_len=1,
                seed= seed if seed != -1 else None
            )
            print(f"loaded {len(ds)} test examples")

            with torch.no_grad():
                for x,gt,nameee in ds:
                    x = x.unsqueeze(0).to(model.device)
                    y = model(x)
                    y = y.cpu().numpy()
                    y =y[0][0]
                    test_gt.append(gt)
                    test_labels.append(y)
                    test_names.append(nameee)

            #compute score for test set
            test_labels = torch.tensor(test_labels).to(model.device)
            test_gt = torch.tensor(test_gt).to(model.device)
            test_names = np.array(test_names)
            test_labels = test_labels.view(-1)

            plcc = model.pearson_corr_coef(test_labels, test_gt)
            spearman = model.spearman_corr_coef(test_labels, test_gt)
            mse_log = model.mse_log(test_labels, test_gt)
            rmse = torch.sqrt(mse_log)

            print(f"test_set{stage}_plcc: {plcc}")
            print(f"test_set{stage}_spearman: {spearman}")
            print(f"test_set{stage}_rmse: {rmse}")

            #save to wandb
            wandb.log({f"final_test_set{stage}_plcc": plcc})
            wandb.log({f"final_test_set{stage}_spearman": spearman})
            wandb.log({f"final_test_set{stage}_rmse": rmse})
            wandb.log({f"final_test_set{stage}_score": (plcc + spearman) / 2})

            scores.append((plcc + spearman) / 2)



            print(f"predicted {len(test_labels)} labels for {name}")
            print(f'len test_names {len(test_names)}')
            print(f'len test_labels {len(test_labels)}')

            #save to file with  Test1_preds.txt, Test2_preds.txt, Test3_preds.txt
            #name, label
            with open(os.path.join(resultsdir, 'Test'+stage+'_preds.txt'), 'w') as f:
                for i in range(len(test_names)):
                    f.write(f"{test_names[i]},{test_labels[i]}\n")

            print(f"saved {len(test_labels)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds.txt')}")

        print(f"final_score: {sum(scores)/3}")
        wandb.log({"final_final_score": sum(scores)/3})


        print("done")

    else:
        print("not rank 0 skipping predictions")
