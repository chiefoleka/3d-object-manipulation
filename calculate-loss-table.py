import json
import os

# Add the files to plot and the respective models in the arrays below
files_train = []
files_test = []
files_name = [
    # 'BasicModel',
    # 'BasicModelNorm',
    # 'BasicModelLarge',
    # 'BasicModelNormLarge',
]
epochs = 1000 # set this value to the epoch the models trained on


def calculate_total_losses(data, total_epochs):
    total_losses = []
    total_objects = len(data)
    for i in range(total_epochs):
        epoch_loss = 0
        for obj in data:
            epoch_loss += obj[i]["agg_loss"]

        total_losses.append(epoch_loss / total_objects)

    return total_losses


def extract_losses_at_epoch(data, epoch):
    res = []

    for filename, obj in data.items():
        loss = obj[epoch - 1]["agg_loss"]
        res.append({"filename": filename, "loss": loss})

    return res


csv_epoch_800 = {"mean_loss": []}
csv_epoch_1000 = {"mean_loss": []}

for i in range(len(files_test)):
    with open(os.path.join("outputs", f"{files_test[i]}"), "r") as f:
        data = json.load(f)
        res_800 = extract_losses_at_epoch(data, 800)
        res_1000 = extract_losses_at_epoch(data, 1000)

        for obj in res_800:
            filename = obj["filename"]
            loss = obj["loss"]

            if filename not in csv_epoch_800:
                csv_epoch_800[filename] = []
            csv_epoch_800[filename].append(loss)
        mean_loss_800 = sum([x["loss"] for x in res_800]) / len(res_800)
        csv_epoch_800["mean_loss"].append(mean_loss_800)

        for obj in res_1000:
            filename = obj["filename"]
            loss = obj["loss"]

            if filename not in csv_epoch_1000:
                csv_epoch_1000[filename] = []
            csv_epoch_1000[filename].append(loss)

        mean_loss_1000 = sum([x["loss"] for x in res_1000]) / len(res_1000)
        csv_epoch_1000["mean_loss"].append(mean_loss_1000)

with open("final_result_800.csv", "w") as f:
    model_names_csv = ",".join(files_name)
    f.write(f"Object name,{model_names_csv}\n")
    for filename, loss in csv_epoch_800.items():
        loss_csv = ",".join([str(x) for x in loss])
        f.write(f"{filename},{loss_csv}\n")

with open("final_result_1000.csv", "w") as f:
    model_names_csv = ",".join(files_name)
    f.write(f"Object name {model_names_csv}\n")
    for filename, loss in csv_epoch_1000.items():
        loss_csv = ",".join([str(x) for x in loss])
        f.write(f"{filename},{loss_csv}\n")
