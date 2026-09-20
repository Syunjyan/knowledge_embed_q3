import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSOLVER_ROOT = Path(
    os.environ.get(
        "TRANSOLVER_ROOT",
        ROOT / "third_party" / "Transolver" / "PDE-Solving-StandardBenchmark",
    )
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRANSOLVER_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default="0")
_pre_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import numpy as np
import scipy.io as scio
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from embed.official_physics import lock_scale, ns_mean_conservation
from embed.run_utils import write_summary

parser = argparse.ArgumentParser("Training Transformer")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--model", type=str, default="Transolver_2D")
parser.add_argument("--n-hidden", type=int, default=64, help="hidden dim")
parser.add_argument("--n-layers", type=int, default=3, help="layers")
parser.add_argument("--n-heads", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--gpu", type=str, default="0", help="GPU index to use")
parser.add_argument("--max_grad_norm", type=float, default=None)
parser.add_argument("--downsample", type=int, default=1)
parser.add_argument("--mlp_ratio", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--unified_pos", type=int, default=0)
parser.add_argument("--ref", type=int, default=8)
parser.add_argument("--slice_num", type=int, default=32)
parser.add_argument("--eval", type=int, default=0)
parser.add_argument("--save_name", type=str, default="ns_2d_UniPDE")
parser.add_argument("--data_path", type=str, default=str(ROOT / "data" / "official"))
parser.add_argument("--group", type=str, default="A")
parser.add_argument("--unlock", type=float, default=0.1)
parser.add_argument("--lock", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--summary_json", type=str, default="")
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

data_path = args.data_path + "/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
ntrain = 1000
ntest = 200
T_in = 10
T = 10
step = 1
eval = args.eval
save_name = args.save_name


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_mat(path):
    try:
        return scio.loadmat(path)
    except NotImplementedError:
        import h5py

        out = {}
        with h5py.File(path, "r") as f:
            out["u"] = np.array(f["u"]).T
        return out


def main():
    r = args.downsample
    h = int(((64 - 1) / r) + 1)
    data = load_mat(data_path)
    print(data["u"].shape)
    train_a = data["u"][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data["u"][:ntrain, ::r, ::r, T_in : T + T_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)

    test_a = data["u"][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data["u"][-ntest:, ::r, ::r, T_in : T + T_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos_train, train_a, train_u),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos_test, test_a, test_u),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print("Dataloading is over.")
    print("group", args.group, "NS C/D use mean-vorticity conservation, not strong-form residual")

    model = get_model(args).Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=T_in,
        out_dim=1,
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=args.unified_pos,
        H=h,
        W=h,
    ).cuda()

    finetune = False
    if args.group.upper() in {"C", "D", "CONS"}:
        a_dir = Path(args.ckpt_dir).resolve().parent / "ns_A"
        summary_p = a_dir / "summary.json"
        ckpt_p = a_dir / "ns_A.pt"
        if summary_p.exists() and ckpt_p.exists():
            meta = json.loads(summary_p.read_text())
            if int(meta.get("epoch", 0)) >= int(meta.get("epochs", 500)) - 2:
                print("finetune from finished ns_A", ckpt_p, "epochs->80")
                model.load_state_dict(torch.load(str(ckpt_p), map_location="cpu"))
                model.cuda()
                finetune = True
                args.epochs = min(int(args.epochs), 80)
                args.unlock = max(float(args.unlock), 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    myloss = TestLoss(size_average=False)
    history = []

    if eval:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, save_name + ".pt")), strict=False)
        model.eval()
        test_l2_full = 0
        with torch.no_grad():
            for x, fx, yy in test_loader:
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                bsz = x.shape[0]
                for t in range(0, T, step):
                    im = model(x, fx=fx)
                    fx = torch.cat((fx[..., step:], im), dim=-1)
                    pred = im if t == 0 else torch.cat((pred, im), -1)
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
        print(test_l2_full / ntest)
        write_summary(args.summary_json, {"case": "ns", "group": args.group, "rel_err": test_l2_full / ntest})
        return

    for ep in range(args.epochs):
        model.train()
        train_l2_step = 0
        train_l2_full = 0
        pde_acc = 0
        n_pde = 0
        phys_w = lock_scale(ep, args.epochs, args.group, args.unlock, args.lock)
        for x, fx, yy in train_loader:
            loss = 0
            pde = fx.new_zeros(())
            x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
            bsz = x.shape[0]
            for t in range(0, T, step):
                y = yy[..., t : t + step]
                im = model(x, fx=fx)
                loss = loss + myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                if phys_w > 0:
                    pde = pde + ns_mean_conservation(im, fx[..., -1])
                    n_pde += 1
                pred = im if t == 0 else torch.cat((pred, im), -1)
                fx = torch.cat((fx[..., step:], y), dim=-1)
            if phys_w > 0:
                loss = loss + phys_w * pde
            train_l2_step += loss.item() if phys_w == 0 else (loss - phys_w * pde).item()
            train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            pde_acc += float(pde.detach())
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        test_l2_step = 0
        test_l2_full = 0
        model.eval()
        with torch.no_grad():
            for x, fx, yy in test_loader:
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                bsz = x.shape[0]
                for t in range(0, T, step):
                    y = yy[..., t : t + step]
                    im = model(x, fx=fx)
                    loss = loss + myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    pred = im if t == 0 else torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., step:], im), dim=-1)
                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        print(
            "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f} , pde:{:.5f} , phys_w:{:.4f}".format(
                ep,
                train_l2_step / ntrain / (T / step),
                train_l2_full / ntrain,
                test_l2_step / ntest / (T / step),
                test_l2_full / ntest,
                pde_acc / max(len(train_loader), 1),
                phys_w,
            )
        )
        history.append(
            {
                "epoch": ep,
                "train_full": train_l2_full / ntrain,
                "test_full": test_l2_full / ntest,
                "phys_w": phys_w,
            }
        )
        write_summary(
            args.summary_json,
            {
                "case": "ns",
                "group": args.group,
                "phys": "none" if args.group.upper() == "A" else "mean_conservation",
                "finetune": finetune,
                "epoch": ep,
                "epochs": args.epochs,
                "rel_err": test_l2_full / ntest,
                "test_step": test_l2_step / ntest / (T / step),
                "phys_w": phys_w,
                "history": history,
            },
        )
        if ep % 100 == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            print("save model")
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, save_name + ".pt"))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print("save model")
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, save_name + ".pt"))


if __name__ == "__main__":
    main()
