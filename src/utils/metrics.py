import torch
from torch.nn import functional
from torch.cuda import amp

def metrics(model, loader, device):
    cost = 0.0
    tp = tn = fp = fn = 0
    model = model.to(device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long).squeeze(1)
            
            with amp.autocast():
                scores = model(x)
                cost += functional.cross_entropy(scores, y).item()
                
                preds = torch.max(scores, dim=1)[1]
                tp += ((preds == 1) & (y == 1)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = (tp) / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    val_cost = cost / len(loader)
    
    return val_cost, acc, dice, iou, tp, tn, fp, fn