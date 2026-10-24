import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import accuracy_score
from collections import defaultdict
from model import build_model,DEVICE

DATASET_NAME="rose_petal"
DATA_DIR=os.path.join("Data","preprocessed",DATASET_NAME)
MODEL_DIR=os.path.join("model",DATASET_NAME)
IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0001
PATIENCE=10
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

def get_group(path):
    stem=os.path.splitext(os.path.basename(path))[0]
    if "_aug" in stem:
        stem=stem.rsplit("_aug",1)[0]
    return os.path.basename(os.path.dirname(path)),stem

def make_splits(dataset):
    grouped=defaultdict(list)
    for index,(path,label) in enumerate(dataset.samples):
        grouped[(label,get_group(path))].append(index)
    by_class=defaultdict(list)
    for label,group in grouped:
        by_class[label].append(group)
    train_groups=[]
    val_groups=[]
    test_groups=[]
    rng=random.Random(SEED)
    for label,groups in sorted(by_class.items()):
        rng.shuffle(groups)
        count=len(groups)
        if count<3:
            raise ValueError(f"Class {dataset.classes[label]} needs at least 3 original images.")
        n_test=max(1,round(count*0.30))
        n_val=max(1,round((count-n_test)*0.15))
        n_val=min(n_val,count-n_test-1)
        test_groups.extend((label,g) for g in groups[:n_test])
        val_groups.extend((label,g) for g in groups[n_test:n_test+n_val])
        train_groups.extend((label,g) for g in groups[n_test+n_val:])
    group_sets=[]
    for selected in (train_groups,val_groups,test_groups):
        group_sets.append([i for label,g in selected for i in grouped[(label,g)]])
    train_idx,val_idx,test_idx=group_sets
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("A dataset split is empty. Add more original images per class.")
    return train_idx,val_idx,test_idx

def save_splits(dataset,splits):
    result={}
    for name,indices in zip(("train","validation","test"),splits):
        result[name]=[os.path.relpath(dataset.samples[i][0],DATA_DIR) for i in indices]
    with open(os.path.join(MODEL_DIR,"splits.json"),"w") as file:
        json.dump(result,file,indent=2)

def main():
    reset_random()
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Run preprocess.py first: {DATA_DIR}")
    os.makedirs(MODEL_DIR,exist_ok=True)
    transform=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    dataset=datasets.ImageFolder(DATA_DIR,transform=transform)
    if len(dataset.classes)<2:
        raise ValueError("At least two classes are required.")
    train_idx,val_idx,test_idx=make_splits(dataset)
    splits=(train_idx,val_idx,test_idx)
    save_splits(dataset,splits)
    with open(os.path.join(MODEL_DIR,"class_mapping.json"),"w") as file:
        json.dump({i:name for i,name in enumerate(dataset.classes)},file,indent=2)
    generator=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(Subset(dataset,train_idx),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,generator=generator)
    val_loader=DataLoader(Subset(dataset,val_idx),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    model=build_model(num_classes=len(dataset.classes),pretrained=True).to(DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3)
    history={"train_loss":[],"train_accuracy":[],"val_loss":[],"val_accuracy":[]}
    best_loss=float("inf")
    patience_count=0
    for epoch in range(EPOCHS):
        model.train()
        total_loss=0.0
        train_true=[]
        train_pred=[]
        for images,labels in train_loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=criterion(logits,labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*labels.size(0)
            train_true.extend(labels.cpu().tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().tolist())
        train_loss=total_loss/len(train_idx)
        train_acc=accuracy_score(train_true,train_pred)
        model.eval()
        val_loss_sum=0.0
        val_true=[]
        val_pred=[]
        with torch.no_grad():
            for images,labels in val_loader:
                images,labels=images.to(DEVICE),labels.to(DEVICE)
                logits=model(images)
                loss=criterion(logits,labels)
                val_loss_sum+=loss.item()*labels.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
        val_loss=val_loss_sum/len(val_idx)
        val_acc=accuracy_score(val_true,val_pred)
        scheduler.step(val_loss)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss<best_loss:
            best_loss=val_loss
            patience_count=0
            torch.save({"model_state_dict":model.state_dict(),"num_classes":len(dataset.classes),"classes":dataset.classes,"dataset_name":DATASET_NAME,"epoch":epoch+1,"best_val_loss":best_loss},os.path.join(MODEL_DIR,"best_model.pth"))
        else:
            patience_count+=1
        with open(os.path.join(MODEL_DIR,"history.json"),"w") as file:
            json.dump(history,file,indent=2)
        if patience_count>=PATIENCE:
            print("Early stopping.")
            break
    print(f"Best model saved: {os.path.join(MODEL_DIR,'best_model.pth')}")

if __name__=="__main__":
    main()