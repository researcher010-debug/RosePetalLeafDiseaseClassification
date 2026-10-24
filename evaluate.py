import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from model import build_model,DEVICE

DATASET_NAME="rose_petal"
DATA_DIR=os.path.join("Data","preprocessed",DATASET_NAME)
MODEL_DIR=os.path.join("model",DATASET_NAME)
RESULT_DIR=os.path.join(MODEL_DIR,"evaluation")
IMAGE_SIZE=224
BATCH_SIZE=32
SEED=42
NUM_WORKERS=0

def reset_random(seed=SEED):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def load_splits(dataset):
    path=os.path.join(MODEL_DIR,"splits.json")
    with open(path) as file:
        saved=json.load(file)
    path_to_index={os.path.relpath(path,DATA_DIR):i for i,(path,_) in enumerate(dataset.samples)}
    splits={}
    for name in ("train","validation","test"):
        missing=[p for p in saved[name] if p not in path_to_index]
        if missing:
            raise ValueError(f"Dataset changed since training; missing {len(missing)} files in {name} split.")
        splits[name]=[path_to_index[p] for p in saved[name]]
    return splits

def evaluate(model,loader,classes,split):
    model.eval()
    targets=[]
    predictions=[]
    total_loss=0.0
    criterion=torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images,labels in loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            logits=model(images)
            total_loss+=criterion(logits,labels).item()*labels.size(0)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    metrics={
        "loss":float(total_loss/len(loader.dataset)),
        "accuracy":float(accuracy_score(targets,predictions)),
        "precision_macro":float(precision_score(targets,predictions,average="macro",zero_division=0)),
        "recall_macro":float(recall_score(targets,predictions,average="macro",zero_division=0)),
        "f1_macro":float(f1_score(targets,predictions,average="macro",zero_division=0)),
        "precision_weighted":float(precision_score(targets,predictions,average="weighted",zero_division=0)),
        "recall_weighted":float(recall_score(targets,predictions,average="weighted",zero_division=0)),
        "f1_weighted":float(f1_score(targets,predictions,average="weighted",zero_division=0))
    }
    report=classification_report(targets,predictions,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    with open(os.path.join(RESULT_DIR,f"{split}_classification_report.json"),"w") as file:
        json.dump(report,file,indent=2)
    cm=confusion_matrix(targets,predictions,labels=list(range(len(classes))))
    display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
    display.plot(xticks_rotation=45,values_format="d")
    plt.title(f"{split.title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,f"{split}_confusion_matrix.png"),dpi=300)
    plt.close()
    return metrics

def plot_history():
    path=os.path.join(MODEL_DIR,"history.json")
    with open(path) as file:
        history=json.load(file)
    plots=[
        ("train_loss","val_loss","Loss","Training and Validation Loss","loss_curve.png"),
        ("train_accuracy","val_accuracy","Accuracy","Training and Validation Accuracy","accuracy_curve.png")
    ]
    for train_key,val_key,ylabel,title,filename in plots:
        plt.figure(figsize=(8,5))
        plt.plot(history[train_key],label="Training")
        plt.plot(history[val_key],label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR,filename),dpi=300)
        plt.close()

def main():
    reset_random()
    os.makedirs(RESULT_DIR,exist_ok=True)
    checkpoint_path=os.path.join(MODEL_DIR,"best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Train the model first: {checkpoint_path}")
    transform=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    dataset=datasets.ImageFolder(DATA_DIR,transform=transform)
    checkpoint=torch.load(checkpoint_path,map_location=DEVICE,weights_only=False)
    classes=checkpoint["classes"]
    if classes!=dataset.classes:
        raise ValueError("Class mapping differs from the saved checkpoint.")
    model=build_model(num_classes=len(classes),pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    splits=load_splits(dataset)
    results={}
    for name in ("train","test"):
        loader=DataLoader(Subset(dataset,splits[name]),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
        results[name]=evaluate(model,loader,classes,name)
    with open(os.path.join(RESULT_DIR,"metrics.json"),"w") as file:
        json.dump(results,file,indent=2)
    plot_history()
    print(json.dumps(results,indent=2))
    print(f"Results saved to {RESULT_DIR}")

if __name__=="__main__":
    main()