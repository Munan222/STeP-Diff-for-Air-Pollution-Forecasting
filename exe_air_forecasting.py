import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Forecasting

# from dataset_forecasting import get_dataloader
from air_forecasting import get_dataloader

from utils import train, evaluate, MYevaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
# parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--unconditional", action="store_true")
# parser.add_argument("--unconditional", default='False')
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250115_111847") # Nan Jing mobile
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250115_141619") # Chang Shu mobile
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_145256") # NJ mobile with deeponet
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_145417") # CS mobile with deeponet
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_155101") # NJ mobile with deeponet add12h
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_155227") # CS mobile with deeponet add12h

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_215402") # NJ with PDE
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_215553") # CS with PDE

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_220448") # NJ DeepONet PDE
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250116_220714") # CS DeepONet PDE

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_192946") # NJ PDE 4city
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_195208") # NJ DeepONetPDE 4city
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_205420") # NJ DeepONet_PDEloss

'''? best one'''
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_211103") # NJ CSDI_PDEloss DeepONet
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250127_224024") # NJ CSDI_PDEloss DeepONet



# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250118_215113") # NJ CSDI_PDEloss

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_193007") # CS PDE 4city
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_195330") # CS DeepONet PDE 4city
# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_205602") # CS DeepONet_PDEloss


# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250117_211319") # CS CSDI_PDEloss DeepONet

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250118_215333") # CS CSDI_PDEloss

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250121_084818") # NJ CSDI_PDEloss{(t_t-1)^2}

# parser.add_argument("--modelfolder", type=str, default="forecasting_electricity_20250125_223110") # NJ CSDI noise

parser.add_argument("--modelfolder", type=str, default="") # omega 0.25,0.5,2,4,8



parser.add_argument("--nsample", type=int, default=100)
# parser.add_argument("--nsample", type=int, default=2)


args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# if args.datatype == 'electricity':
#     target_dim = 370

# target_dim = 100
target_dim = 200
# target_dim = 300



config["model"]["is_unconditional"] = args.unconditional


print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
# train_loader, valid_loader, test_loader, scaler, mean_scaler, Stations_loader = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim


print("Test:")

# evaluate(
#     model,
#     test_loader,
#     nsample=args.nsample,
#     scaler=scaler,
#     mean_scaler=mean_scaler,
#     foldername=foldername,
# )

# MYevaluate(
#     model,
#     test_loader,
#     nsample=args.nsample,
#     scaler=scaler,
#     mean_scaler=mean_scaler,
#     foldername=foldername,
# )

MYevaluate(
    model,
    test_loader,
    # Stations_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
